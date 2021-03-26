import logging
import time
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri

import src.utils as ut

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def generate_df_jobs(list_cutoff: list,
                     run_name: str,
                     algorithm: str,
                     refined_data_specific_path: str
                     ):
    """Generates an empty pd.DataFrame `df_jobs`

    Given arguments for cutoffs, run_name & paths, returns a pd.DataFrame `df_jobs` on which sagemaker utils can iterate
    to handle regarding training & inference.

    Args:
        list_cutoff: A list of integers in format YYYYWW (ISO 8601)
        run_name: A string describing the training & inference run name (for readability)
        algorithm: The used algorithm name (for path purpose)
        refined_data_refined_path: A string pointing to a S3 path (URI format) for input data with trailing slash

    Returns:
        A pandas DataFrame containing all information for each cutoff to handle training & inference
    """

    df_jobs = pd.DataFrame()
    run_suffix = _get_timestamp()
    # Global
    df_jobs['cutoff'] = list_cutoff
    df_jobs['base_job_name'] = [f'{run_name}-{c}' for c in df_jobs['cutoff']]

    # Training
    df_jobs['train_path'] = [f'{refined_data_specific_path}{run_name}/{algorithm}/{n}/input/train_{c}{run_suffix}.json' for (c, n) in zip(df_jobs['cutoff'], df_jobs['base_job_name'])]
    df_jobs['training_job_name'] = np.nan
    df_jobs['training_status'] = 'NotStarted'

    # Inference
    df_jobs['predict_path'] = [f'{refined_data_specific_path}{run_name}/{algorithm}/{n}/input/predict_{c}{run_suffix}.json' for (c, n) in zip(df_jobs['cutoff'], df_jobs['base_job_name'])]
    df_jobs['transform_job_name'] = np.nan
    df_jobs['transform_status'] = 'NotStarted'

    # Serialized model path
    df_jobs['model_path'] = [f'{refined_data_specific_path}{run_name}/{algorithm}/{n}/model/' for (c, n) in zip(df_jobs['cutoff'], df_jobs['base_job_name'])]

    # Jsonline prediction file output path
    df_jobs['output_path'] = [f'{refined_data_specific_path}{run_name}/{algorithm}/{n}/output/' for (c, n) in zip(df_jobs['cutoff'], df_jobs['base_job_name'])]

    logger.debug(f"df_jobs created for cutoffs {list_cutoff}")

    return df_jobs


def _get_timestamp():
    timestamp_suffix = "-" + datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]

    return timestamp_suffix


def import_sagemaker_params(environment: str,
                            ) -> dict:
    """Handler to import sagemaker configuration from YML file

    Args:
        environment (str): Set of parameters on which to load the parameters

    Returns:
        A dictionary with all parameters for sagemaker training & inference
    """
    params_full_path = f"config/{environment}.yml"
    params = ut.read_yml(params_full_path)

    sagemaker_params = {'train_instance_type': params['modeling_parameters']['train_instance_type'],
                        'train_instance_count': params['modeling_parameters']['train_instance_count'],
                        'train_max_instances': params['modeling_parameters']['train_max_instances'],
                        'train_use_spot_instances': params['modeling_parameters']['train_use_spot_instances'],
                        'transform_instance_type': params['modeling_parameters']['transform_instance_type'],
                        'transform_instance_count': params['modeling_parameters']['transform_instance_count'],
                        'transform_max_instances': params['modeling_parameters']['transform_max_instances'],
                        'role': params['modeling_parameters']['role'],
                        'image_name_label': params['modeling_parameters']['image_name_label'],
                        'tags': [params['modeling_parameters']['tags']],
                        'hyperparameters': params['modeling_parameters']['hyperparameters']
                        }

    return sagemaker_params


class SagemakerHandler:
    """
    Sagemaker API handler. Allows for training and transform jobs.
    """

    def __init__(self,
                 run_name: str,
                 df_jobs,
                 train_instance_type: str,
                 train_instance_count: int,
                 train_max_instances: int,
                 train_use_spot_instances: bool,
                 transform_instance_type: str,
                 transform_instance_count: int,
                 transform_max_instances: int,
                 max_concurrent_transforms: int,
                 role: str,
                 image_name_label: str,
                 tags: dict,
                 hyperparameters: dict
                 ):
        # tests
        assert isinstance(run_name, (str))
        assert isinstance(df_jobs, pd.DataFrame)

        # Attributes
        self.run_name = run_name
        self.df_jobs = df_jobs
        self.train_instance_type = train_instance_type
        self.train_instance_count = train_instance_count
        self.train_max_instances = train_max_instances
        self.train_use_spot_instances = train_use_spot_instances
        self.transform_instance_type = transform_instance_type
        self.transform_instance_count = transform_instance_count
        self.transform_max_instances = transform_max_instances
        self.max_concurrent_transforms = max_concurrent_transforms
        self.role = role
        self.image_name_label = image_name_label
        self.tags = tags
        self.hyperparameters = hyperparameters

        # Timestamp suffix definition
        self.run_suffix = _get_timestamp()

        if self.train_use_spot_instances:
            self.train_max_wait = 3600
            self.train_max_run = 3600
        else:
            self.train_max_wait = None
            self.train_max_run = 3600 * 5

        self.sagemaker_session = sagemaker.Session()
        self.image_name = get_image_uri(boto3.Session().region_name, self.image_name_label)

    def launch_training_jobs(self):
        job_type = 'training'

        logger.info(f"Launching {self.df_jobs.shape[0]} training jobs on AWS Sagemaker...")
        while set(self.df_jobs['training_status'].unique()) - {'Failed', 'Completed', 'Stopped'} != set():

            # Condition to check if the running instances limit is not capped
            if self.df_jobs[self.df_jobs['training_status'].isin(['InProgress', 'Stopping'])].shape[0] < self.train_max_instances:

                # Waiting for jobs status to propagate to Sagemaker API
                time.sleep(10)

                # Identifying jobs to start
                df_jobs_to_start = self._identify_jobs_to_start(self.train_max_instances, job_type)

                # Starting jobs
                for i, row in df_jobs_to_start.iterrows():
                    base_job_name = row['base_job_name']
                    model_path = row['model_path']

                    # Creating the estimator
                    estimator = sagemaker.estimator.Estimator(
                        sagemaker_session=self.sagemaker_session,
                        image_name=self.image_name,
                        role=self.role,
                        tags=self.tags,
                        train_instance_count=self.train_instance_count,
                        train_instance_type=self.train_instance_type,
                        base_job_name=base_job_name,
                        output_path=model_path,  # the output of the estimator is the serialized model
                        train_use_spot_instances=self.train_use_spot_instances,
                        train_max_run=self.train_max_run,
                        train_max_wait=self.train_max_wait
                    )

                    # Setting the hyperparameters
                    estimator.set_hyperparameters(**self.hyperparameters)

                    # Launching the fit
                    logger.debug(f"Starting fit for job {base_job_name}")
                    estimator.fit(inputs={'train': row['train_path']}, wait=False)

                    # fill job name
                    self.df_jobs.loc[i, 'training_job_name'] = estimator.latest_training_job.job_name

                # Update, save & display df_jobs
                self._update_jobs_status(job_type)

            # Waiting for jobs status to propagate to Sagemaker API
            time.sleep(10)
            self._update_jobs_status(job_type)

        # Waiting for jobs status to propagate to Sagemaker API
        time.sleep(10)
        self._update_jobs_status(job_type)

        logger.info("Training jobs finished.")

    def launch_transform_jobs(self):
        job_type = 'transform'

        logger.info(f"Launching {self.df_jobs.shape[0]} inference jobs on AWS Sagemaker...")
        while set(self.df_jobs['transform_status'].unique()) - {'Failed', 'Completed', 'Stopped'} != set():

            # Condition to check if the running instances limit is not capped
            if self.df_jobs[self.df_jobs['transform_status'].isin(['InProgress', 'Stopping'])].shape[0] < self.transform_max_instances:

                # Waiting for jobs status to propagate to Sagemaker API
                time.sleep(10)

                # Identifying jobs to start
                df_jobs_to_start = self._identify_jobs_to_start(self.transform_max_instances, job_type)

                # Starting jobs
                for i, row in df_jobs_to_start.iterrows():

                    base_job_name = row['base_job_name']
                    training_job_name = row['training_job_name']
                    output_path = row['output_path']

                    # Delete old model version if exists
                    try:
                        self.sagemaker_session.delete_model(base_job_name)
                    except:
                        pass

                    # Create new one
                    model = self.sagemaker_session.create_model_from_job(
                        training_job_name=training_job_name,
                        name=base_job_name,
                        role=self.role,
                        tags=self.tags
                    )

                    # Define transformer
                    transformer = sagemaker.transformer.Transformer(
                        model_name=base_job_name,
                        instance_count=self.transform_instance_count,
                        instance_type=self.transform_instance_type,
                        strategy='SingleRecord',
                        assemble_with='Line',
                        max_concurrent_transforms=self.max_concurrent_transforms,
                        base_transform_job_name=base_job_name,
                        output_path=output_path,
                        sagemaker_session=self.sagemaker_session,
                        tags=self.tags
                    )

                    # Forecast
                    logger.debug("Launching inference job {training_job_name}")
                    transformer.transform(data=row['predict_path'], split_type='Line')

                    # Fill job name
                    self.df_jobs.loc[i, 'transform_job_name'] = transformer.latest_transform_job.name

                # Update, save & display df_jobs
                self._update_jobs_status(job_type)

            # Waiting for jobs status to propagate to Sagemaker API
            time.sleep(10)
            self._update_jobs_status(job_type)

        # Waiting for jobs status to propagate to Sagemaker API
        time.sleep(10)
        self._update_jobs_status(job_type)

        logger.info("Transform jobs finished.")

    def _identify_jobs_to_start(self, max_running_instances, job_type):
        """Returns the jobs to start by analyzing the Sagemaker jobs monitoring dataframe
        and the maximum number of concurrent jobs authorized by AWS
        """

        status_col = f'{job_type}_status'
        nb_running = self.df_jobs[self.df_jobs[status_col].isin(['InProgress', 'Stopping'])].shape[0]
        nb_to_start = max_running_instances - nb_running

        df_jobs_to_start = self.df_jobs[self.df_jobs[status_col].isin(['NotStarted'])].iloc[:nb_to_start].copy()

        return df_jobs_to_start

    def _update_jobs_status(self, job_type='training'):
        """Updates the Sagemaker jobs monitoring dataframe
        Will ignore :
        * jobs in a `Completed` status
        * jobs that have not been started yet
        """
        job_name_col = f'{job_type}_job_name'
        status_col = f'{job_type}_status'

        # For each job in df_jobs
        for i, row in self.df_jobs.iterrows():
            # If the job has already started
            if not (pd.isna(row[job_name_col]) or row[status_col] == 'Completed'):
                # Get current status
                if job_type == 'training':
                    status = self.sagemaker_session.describe_training_job(row[job_name_col])['TrainingJobStatus']
                if job_type == 'transform':
                    status = self.sagemaker_session.describe_transform_job(row[job_name_col])['TransformJobStatus']
                if job_type == 'tuning':
                    analytics = sagemaker.HyperparameterTuningJobAnalytics(row[job_name_col])
                    status = analytics.description()['HyperParameterTuningJobStatus']
                # Update status
                self.df_jobs.loc[i, status_col] = status

