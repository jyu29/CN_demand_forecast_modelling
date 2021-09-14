import logging

import pandas as pd
import numpy as np

from src.utils import (from_uri, read_jsonline_s3, read_multipart_parquet_s3, write_df_to_parquet_on_s3)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)



def read_format_deepar_forecast(input_path, output_path):
    """Read and format Sagemaker Deepar jsonline outputs

    Read the input jsonline file to bring back the model ids (not available in the output file), and
    format outputs to a readable Dataframe format, including only the point estimate 'mean' forecasts.
    
    Args:
        input_path (str): String path to the deepar jsonline input file
        output_path (str): String path to the deepar jsonline output file

    Returns:
        forecast (pd.DataFrame): Formatted deepar forecast dataframe
    """
    logger.debug("Read & format deepar forecast...")
    
    model_ids = read_jsonline_s3(*from_uri(input_path))['model_id'].values
    forecast = read_jsonline_s3(*from_uri(output_path)).set_index(model_ids)
    forecast = pd.concat([forecast, forecast['quantiles'].apply(pd.Series)], axis=1)
    forecast.drop(columns=['quantiles'], inplace=True)

    horizon = len(forecast.iloc[0]['mean'])

    forecast = forecast.reset_index().rename(columns={'index': 'model_id'})
    forecast = forecast.apply(pd.Series.explode)
    forecast = forecast.astype(float).clip(lower=0).round().astype(int)

    forecast['forecast_step'] = list(range(1, horizon + 1)) * forecast['model_id'].nunique()
    forecast.rename(columns={'0.5': 'forecast'}, inplace=True)
    forecast = forecast[['model_id', 'forecast_step', 'forecast']]
    
    return forecast


def read_format_forecast(df_jobs_cutoff, short_term_algorithm, long_term_algorithm):
    """Read and format short and long term algorithm's forecasts

    Args:
        df_jobs_cutoff (pd.DataFrame): Run jobs information for a unique cutoff
        short_term_algorithm (str): The short-term forecast algorithm name
        long_term_algorithm (str): The long-term forecast algorithm name

    Returns:
        st_forecast (pd.DataFrame): Formatted short-term forecast dataframe
        lt_forecast (pd.DataFrame): Formatted long-term forecast dataframe
        stacking_output_path (str): The final stacked outputs path, calculated from the long-term forecast output path
    """
    assert df_jobs_cutoff['cutoff'].nunique() == 1, '`df_jobs_cutoff` must contain only one cutoff'

    # Set/calculate usefull paths
    st_input_path = df_jobs_cutoff.loc[df_jobs_cutoff['algorithm'] == short_term_algorithm, 'predict_path'].values[0]
    st_output_path = st_input_path.replace('input', 'output') + '.out'

    lt_input_path = df_jobs_cutoff.loc[df_jobs_cutoff['algorithm'] == long_term_algorithm, 'predict_path'].values[0]
    lt_output_path = lt_input_path.replace('input', 'output') + '.out'

    stacking_output_path = lt_output_path \
        .replace(long_term_algorithm, f'{short_term_algorithm}-{long_term_algorithm}') \
        .replace('json', 'parquet')

    # Read & format short term forecast
    logger.debug("Read & format short term forecast...")

    if short_term_algorithm == 'deepar':
        st_forecast = read_format_deepar_forecast(st_input_path, st_output_path)
    else:
        st_forecast = read_multipart_parquet_s3(*from_uri(st_output_path))
    st_forecast.rename(columns={'forecast': 'st_forecast'}, inplace=True)
        
    # Read & format long term forecast
    logger.debug("Read & format short term forecast...")

    if long_term_algorithm == 'deepar':
        lt_forecast = read_format_deepar_forecast(lt_input_path, lt_output_path)
    else:
        lt_forecast = read_multipart_parquet_s3(*from_uri(lt_output_path))
    lt_forecast.rename(columns={'forecast': 'lt_forecast'}, inplace=True)

    return st_forecast, lt_forecast, stacking_output_path


def compute_stacking(st_forecast, lt_forecast, stacking_start, stacking_stop):
    """Calculate a smooth linear stacking between short & long term forecast during a specified period

    Args:
        st_forecast (pd.DataFrame): Formatted short-term forecast dataframe
        lt_forecast (pd.DataFrame): Formatted long-term forecast dataframe
        stacking_start (int): First stacking forecast step 
        stacking_start (int): Last stacking forecast step

    Returns:
        stacked_forecast (pd.DataFrame): The stacked forecast dataframe
    """
    assert st_forecast['forecast_step'].max() >= stacking_stop, \
        f"Short-term algorithm's forecast horizon is too short to apply stacking. Minimum horizon required: {stacking_stop}"

    assert lt_forecast['forecast_step'].max() >= stacking_stop, \
        f"Long-term algorithm's forecast horizon is too short to apply stacking. Minimum horizon required: {stacking_stop}"

    # Calculate stacking
    nb_stacking_weeks = stacking_stop - stacking_start
    
    stacked_forecast = pd.merge(lt_forecast, st_forecast, how='left')

    stacked_forecast['smooth_weight'] = (stacked_forecast['forecast_step'] - stacking_start) / nb_stacking_weeks

    stacked_forecast['forecast'] = np.where(
        stacked_forecast['forecast_step'] <= stacking_start,
        stacked_forecast['st_forecast'],
        np.where(
            stacked_forecast['forecast_step'] <= stacking_stop,
            stacked_forecast['st_forecast'] * (1 - stacked_forecast['smooth_weight']) + \
            stacked_forecast['lt_forecast'] * stacked_forecast['smooth_weight'],
            stacked_forecast['lt_forecast']
        )
    ).round().astype(int)

    # Format stacked forecast
    stacked_forecast = stacked_forecast[['model_id', 'forecast_step', 'forecast']]

    return stacked_forecast


def calculate_outputs_stacking(df_jobs,
                               short_term_algorithm='deepar',
                               long_term_algorithm='hw',
                               smooth_stacking_range=(10, 16)):
    """Calculate linear smooth stacking between forecasts of 2 algorithms during a specific period, and write it on S3

    Args:
        df_jobs (pd.DataFrame): Previous run jobs information
        short_term_algorithm (str): The short-term forecast algorithm name
        long_term_algorithm (str): The long-term forecast algorithm name
        smooth_stacking_range (tuple): Tuple defining the period, i.e. the first and last forecast step (exluded), 
            during which stacking is applied
            Ex: (10, 16) means 100% of short-term algorithm's forecast until forecast step 10 (included), 
                then an evolving mix of short and long-term algorithm's forecast between forecast step 11 and 15,
                and finally the long term algorithm's forecast from forecast step 16

    """
    assert df_jobs['algorithm'].nunique() == 2, "df_jobs should contains exactly 2 differents algorithms for stacking."
    
    assert all(a in df_jobs['algorithm'].unique() for a in [short_term_algorithm, long_term_algorithm]), \
        f"{short_term_algorithm} & {long_term_algorithm} must the 2 algorithms launched in the previous steps."

    assert (all(isinstance(v, int) for v in smooth_stacking_range)) & \
           (smooth_stacking_range[0] > 1) & \
           (smooth_stacking_range[1] > smooth_stacking_range[0]), \
           "The stacking range should be a tuple of 2 integer respecting (a, b): 1 < a < b."

    list_cutoff = df_jobs['cutoff'].unique()
    stacking_start = smooth_stacking_range[0]
    stacking_stop = smooth_stacking_range[1]

    for cutoff in list_cutoff:

        logger.info(f"Calculate {short_term_algorithm}-{long_term_algorithm} for cutoff {cutoff}...")

        df_jobs_cutoff = df_jobs[df_jobs['cutoff'] == cutoff].copy()

        st_forecast, lt_forecast, stacking_output_path = read_format_forecast(df_jobs_cutoff,
                                                                              short_term_algorithm,
                                                                              long_term_algorithm)

        stacked_forecast = compute_stacking(st_forecast, lt_forecast, stacking_start, stacking_stop)

        write_df_to_parquet_on_s3(stacked_forecast, *from_uri(stacking_output_path), verbose=True)