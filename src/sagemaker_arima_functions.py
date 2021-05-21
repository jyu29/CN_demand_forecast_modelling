import os
import warnings
import pandas as pd

from joblib import Parallel, delayed
from pmdarima import AutoARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import FourierFeaturizer
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
        'fourier_seasonal_period': int(params['fourier_seasonal_period']),
        'fourier_order': int(params['fourier_order']) if params['fourier_order'] != 'None' else None,
        'arima_differencing_order': int(params['arima_differencing_order']) \
                                    if params['arima_differencing_order'] != 'None' else None,
        'arima_criterion': params['arima_criterion'],
        'arima_optimizer': params['arima_optimizer']
    }

    return config_params, hyperparameters


def _fit_predict_ts(df_ts, prediction_length, fourier_seasonal_period, fourier_order, 
                    arima_differencing_order, arima_criterion, arima_optimizer):
    """
    Produces forecasts for a single time series.

    Args:
        df_ts (pd.DataFrame): The timeseries DataFrame
        prediction_length (int): The forecasting horizon
        fourier_seasonal_period (int): The seasonal period used in FourierFeaturizer
        fourier_order (int): The fourier order used in FourierFeaturizer
        arima_differencing_order (int): The ARIMA differentiation parameter (d)
        arima_criterion (str): The information criterion used for auto-ARIMA
        arima_optimizer (str): the optimization method used for auto-ARIMA

    Returns:
        df_forecast (pd.DataFrame): The output forecast DataFrame
    """
    warnings.filterwarnings("ignore")

    df_ts = df_ts.sort_values(['week_id'])
    model_id = df_ts['model_id'].iloc[0]

    # Define model pipeline
    pipe = Pipeline([
        ("fourier", FourierFeaturizer(m=fourier_seasonal_period,
                                      k=fourier_order)),
        ("arima", AutoARIMA(d=arima_differencing_order,
                            seasonal=False, # because we use Fourier
                            information_criterion=arima_criterion,
                            method=arima_optimizer, # default is lbfgs
                            suppress_warnings=True,
                            trace=False,
                            error_action="ignore"))
    ])

    try:
        # Fit
        pipe.fit(df_ts['sales_quantity'])

        # Predict
        forecast = pipe.predict(prediction_length)

    except ValueError:
        raise ValueError(f'Crash on model {model_id}!')

    # Format
    df_forecast = pd.DataFrame()
    df_forecast['forecast'] = forecast
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

    l_df_forecast = Parallel(n_jobs=-1, verbose=1) \
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
