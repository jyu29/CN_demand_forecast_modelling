import os
import time
import logging
import sagemaker

import numpy as np
import pandas as pd

from datetime import datetime
from itertools import product
from src.utils import (is_iso_format, read_yml, check_run_name, check_environment)


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

SUPPORTED_ALGORITHMS = ['deepar', 'arima', 'hw']
CONFIG_PATH = 'config'


def generate_df_jobs(list_cutoff: list,
                     run_name: str,
                     list_algorithm: list,
                     refined_data_specific_path: str
                     ):
    """Generates a pd.DataFrame `df_jobs`

    Given arguments for cutoffs, run_name & algorithms, returns a pd.DataFrame
    `df_jobs` on which sagemaker utils can iterate to handle regarding
    training & inference.

    Args:
        list_cutoff: A list of integers in format YYYYWW (ISO 8601)
        run_name: A string describing the training & inference run name
                    (for readability)
        algorithm: The used algorithm name (for path purpose)
        refined_data_refined_path: A string pointing to a S3 path (URI format)
                    for modeling data with trailing slash

    Returns:
        A pandas DataFrame containing all information for each cutoff to handle training & inference
    """

    assert isinstance(list_cutoff, (list))
    for c in list_cutoff:
        assert isinstance(c, (int))
        assert is_iso_format(c)
    assert isinstance(list_algorithm, (list))
    for a in list_algorithm:
        assert isinstance(a, (str))

    l_dict_job = []
    data_timestamp = _get_timestamp()
    data_path = f'{refined_data_specific_path}{run_name}'

    for algorithm, cutoff in product(list_algorithm, list_cutoff):

        dict_job = {}
        base_job_name = f'{run_name}-{algorithm}-{cutoff}'
        check_run_name(base_job_name, check_reserved_words=False)

        # Set file extension
        file_extension = 'parquet'  # default
        if algorithm == 'deepar':
            file_extension = 'json'

        # Global
        dict_job['algorithm'] = algorithm
        dict_job['cutoff'] = cutoff
        dict_job['base_job_name'] = base_job_name

        # Input
        dict_job['train_path'] = f'{data_path}/{base_job_name}/input/train{data_timestamp}.{file_extension}'
        dict_job['training_job_name'] = np.nan
        dict_job['training_status'] = 'NotStarted'

        dict_job['predict_path'] = f'{data_path}/{base_job_name}/input/predict{data_timestamp}.{file_extension}'
        dict_job['transform_job_name'] = np.nan
        dict_job['transform_status'] = 'NotStarted'

        # Model
        dict_job['model_path'] = f'{data_path}/{base_job_name}/model/'

        # Output
        dict_job['output_path'] = f'{data_path}/{base_job_name}/output/'

        l_dict_job.append(dict_job)

    df_jobs = pd.DataFrame.from_dict(l_dict_job)

    logger.debug(f"df_job created for algorithm {list_algorithm} and cutoffs {list_cutoff}")

    return df_jobs


def _get_timestamp():
    timestamp_suffix = "-" + datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]

    return timestamp_suffix


def import_sagemaker_params(environment: str,
                            algorithm: str) -> dict:
    """Handler to import sagemaker configuration from YML file

    Args:
        environment (str): Set of parameters on which to load the parameters
        algorithm (str): List of algorithm name

    Returns:
        A dictionary with all parameters for sagemaker training & inference
    """
    assert isinstance(environment, str)
    check_environment(environment, CONFIG_PATH)
    assert algorithm in SUPPORTED_ALGORITHMS, \
        f"Algorithm {algorithm} not in list of supported algorithms {SUPPORTED_ALGORITHMS}"

    params_full_path = os.path.join(CONFIG_PATH, f"{environment}.yml")
    params = read_yml(params_full_path)

    # Mandatory params
    sagemaker_params = {
        'role': params['modeling_parameters']['role'],
        'tags': [params['modeling_parameters']['tags']],
        'train_use_spot_instances': params['modeling_parameters']['train_use_spot_instances'],
        'image_name': params['modeling_parameters']['algorithm'][algorithm]['image_name'],
        'hyperparameters': params['modeling_parameters']['algorithm'][algorithm]['hyperparameters'],
        'train_instance_count': params['modeling_parameters']['algorithm'][algorithm]['train_instance_count'],
        'train_instance_type': params['modeling_parameters']['algorithm'][algorithm]['train_instance_type'],
        'train_max_instances': params['modeling_parameters']['algorithm'][algorithm]['train_max_instances']
    }

    # Optionnal params
    if 'transform_instance_count' in params['modeling_parameters']['algorithm'][algorithm]:
        sagemaker_params.update({
            'transform_instance_count': params['modeling_parameters']['algorithm'][algorithm]['transform_instance_count'],
            'transform_instance_type': params['modeling_parameters']['algorithm'][algorithm]['transform_instance_type'],
            'transform_max_instances': params['modeling_parameters']['algorithm'][algorithm]['transform_max_instances'],
            'max_concurrent_transforms': params['modeling_parameters']['algorithm'][algorithm]['max_concurrent_transforms']
        })

    return sagemaker_params


class SagemakerHandler:
    """
    Sagemaker API handler. Allows for training and transform jobs.
    """

    def __init__(self,
                 df_jobs: pd.DataFrame,
                 role: str,
                 tags: dict,
                 train_use_spot_instances: bool,
                 image_name: str,
                 hyperparameters: dict,
                 train_instance_count: int,
                 train_instance_type: str,
                 train_max_instances: int,
                 transform_instance_count: int = None,
                 transform_instance_type: str = None,
                 transform_max_instances: int = None,
                 max_concurrent_transforms: int = None,
                 ):

        # Tests
        assert isinstance(df_jobs, pd.DataFrame)
        assert isinstance(role, str)
        assert isinstance(tags, list)
        assert isinstance(train_use_spot_instances, bool)
        assert isinstance(image_name, str)
        assert isinstance(hyperparameters, dict)
        assert isinstance(train_instance_count, int)
        assert isinstance(train_instance_type, str)
        assert isinstance(train_max_instances, int)

        if transform_instance_count:
            assert isinstance(transform_instance_count, int)
            assert isinstance(transform_instance_type, str)
            assert isinstance(transform_max_instances, int)
            assert isinstance(max_concurrent_transforms, int)

        # Attributes
        self.df_jobs = df_jobs
        self.role = role
        self.tags = tags
        self.train_use_spot_instances = train_use_spot_instances
        self.image_name = image_name
        self.hyperparameters = hyperparameters
        self.train_instance_count = train_instance_count
        self.train_instance_type = train_instance_type
        self.train_max_instances = train_max_instances
        self.sagemaker_session = sagemaker.Session()

        if transform_instance_count:
            self.transform_instance_count = transform_instance_count
            self.transform_instance_type = transform_instance_type
            self.transform_max_instances = transform_max_instances
            self.max_concurrent_transforms = max_concurrent_transforms

        if train_use_spot_instances:
            self.train_max_wait = 3600
            self.train_max_run = 3600
        else:
            self.train_max_wait = None
            self.train_max_run = 3600 * 5

    def launch_training_jobs(self):
        job_type = 'training'

        logger.info(f"Launching {self.df_jobs.shape[0]} training jobs on AWS Sagemaker...")
        while set(self.df_jobs['training_status'].unique()) - {'Failed', 'Completed', 'Stopped'} != set():

            # Condition to check if the running instances limit is not capped
            if self.df_jobs[self.df_jobs['training_status'].isin(['InProgress', 'Stopping'])].shape[0] \
                    < self.train_max_instances:

                # Waiting for jobs status to propagate to Sagemaker API
                time.sleep(10)

                # Identifying jobs to start
                df_jobs_to_start = self._identify_jobs_to_start(self.train_max_instances, job_type)

                # Starting jobs
                for i, job in df_jobs_to_start.iterrows():

                    # Creating the estimator
                    estimator = sagemaker.estimator.Estimator(
                        sagemaker_session=self.sagemaker_session,
                        image_name=self.image_name,
                        role=self.role,
                        tags=self.tags,
                        train_instance_count=self.train_instance_count,
                        train_instance_type=self.train_instance_type,
                        base_job_name=job['base_job_name'],
                        output_path=job['model_path'],  # the output of the estimator is the serialized model
                        train_use_spot_instances=self.train_use_spot_instances,
                        train_max_run=self.train_max_run,
                        train_max_wait=self.train_max_wait
                    )

                    # Setting the hyperparameters
                    job_hyperparameters = self.hyperparameters.copy()

                    if job['algorithm'] in ['arima', 'hw']:
                        job_hyperparameters['input_file_name'] = os.path.basename(job['train_path'])
                        job_hyperparameters['s3_output_path'] = job['output_path']

                    estimator.set_hyperparameters(**job_hyperparameters)

                    # Launching the fit
                    logger.debug(f"Starting fit for job {job['base_job_name']}")
                    estimator.fit(inputs={'train': job['train_path']}, wait=False)

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
            if self.df_jobs[self.df_jobs['transform_status'].isin(['InProgress', 'Stopping'])].shape[0]\
                    < self.transform_max_instances:

                # Waiting for jobs status to propagate to Sagemaker API
                time.sleep(10)

                # Identifying jobs to start
                df_jobs_to_start = self._identify_jobs_to_start(self.transform_max_instances, job_type)

                # Starting jobs
                for i, job in df_jobs_to_start.iterrows():

                    # Delete old model version if exists
                    try:
                        self.sagemaker_session.delete_model(job['base_job_name'])
                    except:  # noqa: E722
                        pass

                    # Create new one
                    model = self.sagemaker_session.create_model_from_job(  # noqa: F841
                        training_job_name=job['training_job_name'],
                        name=job['base_job_name'],
                        role=self.role,
                        tags=self.tags
                    )

                    # Define transformer
                    transformer = sagemaker.transformer.Transformer(
                        model_name=job['base_job_name'],
                        instance_count=self.transform_instance_count,
                        instance_type=self.transform_instance_type,
                        strategy='SingleRecord',
                        assemble_with='Line',
                        max_concurrent_transforms=self.max_concurrent_transforms,
                        base_transform_job_name=job['base_job_name'],
                        output_path=job['output_path'],
                        sagemaker_session=self.sagemaker_session,
                        tags=self.tags
                    )

                    # Forecast
                    logger.debug(f"Launching inference job {job['training_job_name']}")
                    transformer.transform(data=job['predict_path'], split_type='Line')

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
        Will ignore:
        * jobs in a `Completed` status
        * jobs that have not been started yet
        """
        job_name_col = f'{job_type}_job_name'
        status_col = f'{job_type}_status'

        # For each job in df_jobs
        for i, job in self.df_jobs.iterrows():
            # If the job has already started
            if not (pd.isna(job[job_name_col]) or job[status_col] == 'Completed'):
                # Get current status
                if job_type == 'training':
                    status = self.sagemaker_session.describe_training_job(job[job_name_col])['TrainingJobStatus']
                if job_type == 'transform':
                    status = self.sagemaker_session.describe_transform_job(job[job_name_col])['TransformJobStatus']
                if job_type == 'tuning':
                    analytics = sagemaker.HyperparameterTuningJobAnalytics(job[job_name_col])
                    status = analytics.description()['HyperParameterTuningJobStatus']
                # Update status
                self.df_jobs.loc[i, status_col] = status
