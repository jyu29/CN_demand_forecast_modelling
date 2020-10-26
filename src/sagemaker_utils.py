import sagemaker
import pandas as pd
import numpy as np
import src.refined_data_handler as dh


def generate_df_jobs(run_name, cutoffs, bucket, run_input_path):
    # Generating df_jobs
    df_jobs = pd.DataFrame()

    # Global
    df_jobs['cutoff'] = cutoffs
    df_jobs['base_job_name'] = [f'{run_name}-{c}' for c in df_jobs['cutoff']]

    # Training
    df_jobs['train_path'] = [f's3://{bucket}/{run_input_path}train_{c}.json' for c in df_jobs['cutoff']]
    df_jobs['training_job_name'] = np.nan
    df_jobs['training_status'] = 'NotStarted'

    # Inference
    df_jobs['predict_path'] = [f's3://{bucket}/{run_input_path}predict_{c}.json' for c in df_jobs['cutoff']]
    df_jobs['transform_job_name'] = np.nan
    df_jobs['transform_status'] = 'NotStarted'

    return df_jobs


def generate_input_data(row, fs, parameters):
    params = {'cutoff': row['cutoff'],
              'bucket': parameters['buckets']['refined-data'],
              'cat_cols': parameters['functional_parameters']['cat_cols'],
              'min_ts_len': parameters['functional_parameters']['min_ts_len'],
              'prediction_length': parameters['functional_parameters']['prediction_length'],
              'clean_data_path': parameters['paths']['clean_data_path'],
              'run_input_path': parameters['paths']['run_input_path'],
              'hist_rec_method': parameters['functional_parameters']['target_hist_rec_method'],
              'cluster_keys': parameters['functional_parameters']['target_cluster_keys'],
              'patch_covid': parameters['functional_parameters']['patch_covid'],
              'dyn_cols': parameters['functional_parameters']['dyn_cols']
              }
    data_handler = dh.refined_data_handler(params)
    data_handler.import_input_datasets()
    data_handler.generate_deepar_input_data(fs)

    print(f"Cutoff {data_handler.cutoff} : {data_handler.df_train['model'].nunique()} models")


def identify_jobs_to_start(df_jobs, max_running_instances, job_type='training'):
    """Returns the jobs to start by analyzing the Sagemaker jobs monitoring dataframe
    and the maximum number of concurrent jobs authorized by AWS
    """

    status_col = f'{job_type}_status'
    nb_running = df_jobs[df_jobs[status_col].isin(['InProgress', 'Stopping'])].shape[0]
    nb_to_start = max_running_instances - nb_running

    df_jobs_to_start = df_jobs[df_jobs[status_col].isin(['NotStarted'])].iloc[:nb_to_start].copy()

    return df_jobs_to_start


def update_jobs_status(sagemaker_session, df_jobs, job_type='training'):
    """Updates the Sagemaker jobs monitoring dataframe
    Will ignore :
    * jobs in a `Completed` status
    * jobs that have not been started yet
    """
    job_name_col = f'{job_type}_job_name'
    status_col = f'{job_type}_status'

    # For each job in df_jobs
    for i, row in df_jobs.iterrows():
        # If the job has already started
        if not (pd.isna(row[job_name_col]) or row[status_col] == 'Completed'):
            # Get current status
            if job_type == 'training':
                status = sagemaker_session.describe_training_job(row[job_name_col])['TrainingJobStatus']
            if job_type == 'transform':
                status = sagemaker_session.describe_transform_job(row[job_name_col])['TransformJobStatus']
            if job_type == 'tuning':
                analytics = sagemaker.HyperparameterTuningJobAnalytics(row[job_name_col])
                status = analytics.description()['HyperParameterTuningJobStatus']
            # Update status
            df_jobs.loc[i, status_col] = status