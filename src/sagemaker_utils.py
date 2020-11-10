import sagemaker
import boto3
import time
import pandas as pd
import numpy as np
import src.refined_data_handler as dh
import src.utils as ut

from sagemaker.amazon.amazon_estimator import get_image_uri


def generate_df_jobs(run_name, cutoffs, bucket, run_input_path):
    # Generating df_jobs
    df_jobs = pd.DataFrame()

    # Global
    df_jobs['cutoff'] = cutoffs
    df_jobs['base_job_name'] = [f'{run_name}-{c}' for c in df_jobs['cutoff']]

    # Training
    df_jobs['train_path'] = [f's3://{bucket}/{run_input_path}input/train_{c}.json' for c in df_jobs['cutoff']]
    df_jobs['training_job_name'] = np.nan
    df_jobs['training_status'] = 'NotStarted'

    # Inference
    df_jobs['predict_path'] = [f's3://{bucket}/{run_input_path}input/predict_{c}.json' for c in df_jobs['cutoff']]
    df_jobs['transform_job_name'] = np.nan
    df_jobs['transform_status'] = 'NotStarted'

    return df_jobs


def generate_input_data(row, fs, parameters):
    params = {'cutoff': row['cutoff'],
              'run_name': row['base_job_name'],
              'bucket': parameters['buckets']['refined-data'],
              'cat_cols': parameters['functional_parameters']['cat_cols'],
              'min_ts_len': parameters['functional_parameters']['min_ts_len'],
              'prediction_length': parameters['functional_parameters']['prediction_length'],
              'clean_data_path': parameters['paths']['clean_data_path'],
              'refined_data_path': parameters['paths']['refined_path_full'],
              'hist_rec_method': parameters['functional_parameters']['target_hist_rec_method'],
              'cluster_keys': parameters['functional_parameters']['target_cluster_keys'],
              'patch_covid': parameters['functional_parameters']['patch_covid'],
              'dyn_cols': parameters['functional_parameters']['dyn_cols']
              }
    data_handler = dh.refined_data_handler(params)
    data_handler.import_input_datasets()
    data_handler.generate_deepar_input_data(fs)

    print(f"Cutoff {data_handler.cutoff} : {data_handler.df_train['model'].nunique()} models")


class SagemakerHandler:
    """
    Sagemaker API handler. Allows for training and transform jobs.
    """

    def __init__(self, df_jobs, params):
        # tests
        assert type(df_jobs) == pd.DataFrame, "df_jobs wrong format (expected pd.DataFrame)"
        assert df_jobs.shape[0] > 0, "No job to start in `df_jobs`"
        assert 'technical_parameters' in params
        assert 'max_train_instances' in params['technical_parameters']

        # Attributes
        self.df_jobs = df_jobs
        self._columns_display = ['cutoff', 'base_job_name', 'training_job_name', 'training_status']
        self.max_train_instances = params['technical_parameters']['max_train_instances']
        self.role = params['technical_parameters']['role']
        self.image_name_label = params['technical_parameters']['image_name_label']
        self.tags = [params['technical_parameters']['tags']]
        self.train_instance_type = params['technical_parameters']['train_instance_type']
        self.max_train_instances = params['technical_parameters']['max_train_instances']
        self.max_transform_instances = params['technical_parameters']['max_transform_instances']
        self.train_use_spot_instances = params['technical_parameters']['train_use_spot_instances']
        self.bucket = params['buckets']['refined-data']
        self.refined_path = params['paths']['refined_path_full']
        self.train_instance_count = params['technical_parameters']['train_instance_count']
        self.transform_instance_count = params['technical_parameters']['transform_instance_count']
        self.transform_instance_type = params['technical_parameters']['transform_instance_type']

        if self.train_use_spot_instances:
            self.train_max_wait = 3600
            self.train_max_run = 3600
        else:
            self.train_max_wait = None
            self.train_max_run = 3600 * 5

        self.sagemaker_session = sagemaker.Session()
        self.image_name = get_image_uri(boto3.Session().region_name, self.image_name_label)

        self.hyperparameters = params['functional_parameters']['hyperparameters']

    def identify_jobs_to_start(self, max_running_instances, job_type):
        """Returns the jobs to start by analyzing the Sagemaker jobs monitoring dataframe
        and the maximum number of concurrent jobs authorized by AWS
        """

        status_col = f'{job_type}_status'
        nb_running = self.df_jobs[self.df_jobs[status_col].isin(['InProgress', 'Stopping'])].shape[0]
        nb_to_start = max_running_instances - nb_running

        df_jobs_to_start = self.df_jobs[self.df_jobs[status_col].isin(['NotStarted'])].iloc[:nb_to_start].copy()

        return df_jobs_to_start

    def update_jobs_status(self, job_type='training'):
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

    def launch_training_jobs(self):
        job_type = 'training'

        while set(self.df_jobs['training_status'].unique()) - {'Failed', 'Completed', 'Stopped'} != set():

            # Condition to check if the running instances limit is not capped
            if self.df_jobs[self.df_jobs['training_status'].isin(['InProgress', 'Stopping'])].shape[0] < self.max_train_instances:

                # Waiting for jobs status to propagate to Sagemaker API
                time.sleep(10)

                # Identifying jobs to start
                df_jobs_to_start = self.identify_jobs_to_start(self.max_train_instances, job_type)

                # Starting jobs
                for i, row in df_jobs_to_start.iterrows():
                    base_job_name = row['base_job_name']
                    model_path = ut.to_uri(self.bucket, f"{self.refined_path}{base_job_name}/model/")

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
                    estimator.fit(inputs={'train': row['train_path']}, wait=False)

                    # fill job name
                    self.df_jobs.loc[i, 'training_job_name'] = estimator.latest_training_job.job_name

                # Update, save & display df_jobs
                self.update_jobs_status(job_type)
                ut.write_df_to_csv_on_s3(self.df_jobs,
                                         self.bucket,
                                         f"{self.refined_path}{base_job_name}/"+'df_jobs.csv',
                                         verbose=False)

            # Waiting for jobs status to propagate to Sagemaker API
            time.sleep(10)
            self.update_jobs_status(job_type)
            ut.write_df_to_csv_on_s3(self.df_jobs,
                                     self.bucket,
                                     f"{self.refined_path}{base_job_name}/" + 'df_jobs.csv',
                                     verbose=False)

        # Waiting for jobs status to propagate to Sagemaker API
        time.sleep(10)
        self.update_jobs_status(job_type)
        ut.write_df_to_csv_on_s3(self.df_jobs,
                                 self.bucket,
                                 f"{self.refined_path}{base_job_name}/" + 'df_jobs.csv',
                                 verbose=False)

        print('Training done.')

    def launch_transform_jobs(self):
        job_type = 'transform'

        while set(self.df_jobs['transform_status'].unique()) - {'Failed', 'Completed', 'Stopped'} != set():

            # Condition to check if the running instances limit is not capped
            if self.df_jobs[self.df_jobs['transform_status'].isin(['InProgress', 'Stopping'])].shape[0] < self.max_transform_instances:

                # Waiting for jobs status to propagate to Sagemaker API
                time.sleep(10)

                # Identifying jobs to start
                df_jobs_to_start = self.identify_jobs_to_start(self.max_transform_instances, job_type)

                # Starting jobs
                for i, row in df_jobs_to_start.iterrows():

                    base_job_name = row['base_job_name']
                    training_job_name = row['training_job_name']
                    output_path = ut.to_uri(self.bucket, f"{self.refined_path}{base_job_name}/output/")

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
                        base_transform_job_name=base_job_name,
                        output_path=output_path,
                        sagemaker_session=self.sagemaker_session,
                        tags=self.tags
                    )

                    # Forecast
                    transformer.transform(data=row['predict_path'], split_type='Line')

                    # Fill job name
                    self.df_jobs.loc[i, 'transform_job_name'] = transformer.latest_transform_job.name

                # Update, save & display df_jobs
                self.update_jobs_status(job_type)
                ut.write_df_to_csv_on_s3(self.df_jobs,
                                         self.bucket,
                                         f"{self.refined_path}{base_job_name}/" + 'df_jobs.csv',
                                         verbose=False)

            # Waiting for jobs status to propagate to Sagemaker API
            time.sleep(10)
            self.update_jobs_status(job_type)
            ut.write_df_to_csv_on_s3(self.df_jobs,
                                     self.bucket,
                                     f"{self.refined_path}{base_job_name}/" + 'df_jobs.csv',
                                     verbose=False)

        # Waiting for jobs status to propagate to Sagemaker API
        time.sleep(10)
        self.update_jobs_status(job_type)
        ut.write_df_to_csv_on_s3(self.df_jobs,
                                 self.bucket,
                                 f"{self.refined_path}{base_job_name}/" + 'df_jobs.csv',
                                 verbose=False)

        print('Transform job done.')
