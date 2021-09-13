import os
import warnings
import pandas as pd

from joblib import Parallel, delayed
from multiprocessing import cpu_count
from statsmodels.tsa.api import ExponentialSmoothing

from utils import (read_json, from_uri, write_df_to_parquet_on_s3)


def import_parameters(hps_file_path):
    """
    Read all parameters from the input json file path and return specific parameters for configs & model hyperparameters.

    Args:
        hps_file_path (str): The input json file path
    Returns:
        config_params (dict): Dict of config parameters
        hyperparameters (dict): Dict of model hyperparameters
    """
    params = read_json(hps_file_path)

    config_params = {
        'input_file_name': params['input_file_name'],
        's3_output_path': params['s3_output_path']
    }

    hyperparameters = {
        'prediction_length': int(params['prediction_length']),
        'trend': params['trend'] if params['trend'] != 'None' else None,
        'damped_trend': True if params['damped_trend'] == 'True' else False,
        'seasonal': params['seasonal'] if params['seasonal'] != 'None' else None,
        'seasonal_periods': int(params['seasonal_periods']),
        'smoothing_level': float(params['smoothing_level']),
        'smoothing_trend': float(params['smoothing_trend']),
        'smoothing_seasonal': float(params['smoothing_seasonal']),
        'damping_trend': float(params['damping_trend'])
    }

    return config_params, hyperparameters
    

def _fit_predict_ts(df_ts, prediction_length, trend, damped_trend, seasonal, seasonal_periods,
                    smoothing_level, smoothing_trend, smoothing_seasonal, damping_trend):
    """
    Produces forecasts for a single time series.

    Args:
        df_ts (pd.DataFrame): Timeseries DataFrame
        prediction_length (int): Forecasting horizon
        trend (str): Type of trend component
        damped_trend (bool): Should the trend component be damped
        seasonal (str): Type of seasonal component
        seasonal_periods (int): The number of periods in a complete seasonal cycle
        smoothing_level (float): The alpha value of the simple exponential smoothing
        smoothing_trend (float): The beta value of the Holt's trend method
        smoothing_seasonal(float): The gamma value of the holt winters seasonal method
        damping_trend (float): The phi value of the damped method

    Returns:
        df_forecast (pd.DataFrame): The output forecast DataFrame
    """
    warnings.filterwarnings("ignore")

    model_id = df_ts['model_id'].iloc[0]
    df_ts = df_ts[['date', 'sales_quantity']].sort_values('date').set_index('date')

    # Define model
    m = ExponentialSmoothing(
        df_ts + 1.0,
        trend=trend,
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method='estimated',
        use_boxcox=False,
        freq='W-SUN',
        missing='none',
    )

    try:
        # Fit
        fit = m.fit(smoothing_level=smoothing_level,
                    smoothing_trend=smoothing_trend,
                    smoothing_seasonal=smoothing_seasonal,
                    damping_trend=damping_trend,
                    optimized=False)

        # Predict
        forecast = fit.forecast(prediction_length) - 1.0

    except ValueError:
        raise ValueError(f'Crash on model {model_id}!')

    # Format
    df_forecast = pd.DataFrame()
    df_forecast['forecast'] = forecast.values
    df_forecast['forecast'] = df_forecast['forecast'].astype('float').round().astype(int).clip(lower=0)
    df_forecast['model_id'] = model_id
    df_forecast['forecast_step'] = list(range(1, prediction_length + 1))
    df_forecast = df_forecast[['model_id', 'forecast_step', 'forecast']]

    return df_forecast
    
    
def fit_predict_all_ts(df_predict, hyperparameters):
    """
    Produces forecasts for each time series in the input dataframe.

    Args:
        df_predict (pd.DataFrame): DataFrame containing all the time series for which a forecast must be provided
        hyperparameters (dict): The model hyperparameters

    Returns:
        df_forecast (pd.DataFrame): The output forecast DataFrame
    """
    print(f"Launching the fit-predict method in parallel on {df_predict['model_id'].nunique()} time series...")

    l_df_forecast = Parallel(n_jobs=cpu_count()//4, verbose=1) \
                    (delayed(_fit_predict_ts)(df_ts, **hyperparameters) \
                     for _, df_ts in df_predict.groupby(['model_id']))

    df_forecast = pd.concat(l_df_forecast).sort_values(['model_id', 'forecast_step']).reset_index(drop=True)

    return df_forecast


def write_forecast_df_on_s3(df_forecast, s3_output_path, input_file_name):
    """
    Write forecast df on S3.

    Args:
        df_forecast (pd.DataFrame): The output forecast DataFrame
        s3_output_path (str): The output path
        input_file_name (str): The input file name used to create the output one
    """
    bucket, dir_path = from_uri(s3_output_path)
    output_file_name = input_file_name.replace('train', 'predict') + '.out' # to match the output format of Sagemaker DeepAR
    file_path = os.path.join(dir_path, output_file_name)

    write_df_to_parquet_on_s3(df_forecast, bucket, file_path, verbose=True)
