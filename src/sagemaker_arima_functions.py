import pandas as pd

from joblib import Parallel, delayed
from pmdarima import AutoARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import FourierFeaturizer
from src.utils import (week_id_to_date, date_to_week_id, from_uri, write_df_to_parquet_on_s3)


def _fit_predict_ts(df_ts, cutoff, prediction_length, fourier_seasonal_period, fourier_order,
                    arima_differencing_order, arima_criterion, arima_optimizer):
    """
    Produces forecasts for a single time series.

    Args:
        df_ts (pd.DataFrame): The timeseries DataFrame
        cutoff (int): The current cutoff week id
        prediction_length (int): The forecasting horizon
        fourier_seasonal_period (int): The seasonal period used in FourierFeaturizer
        fourier_order (int): The fourier order used in FourierFeaturizer
        arima_differencing_order (int): The ARIMA differentiation parameter (d)
        arima_criterion (str): The information criterion used for auto-ARIMA
        arima_optimizer (str): the optimization method used for auto-ARIMA

    Returns:
        df_forecast (pd.DataFrame): The output forecast DataFrame
    """

    df_ts = df_ts.sort_values(['week_id'])
    model_id = df_ts['model_id'].iloc[0]

    # Define model pipeline
    pipe = Pipeline([
        ("fourier", FourierFeaturizer(m=fourier_seasonal_period,
                                      k=fourier_order)),
        ("arima", AutoARIMA(d=arima_differencing_order,
                            seasonal=False,  # because we use Fourier
                            information_criterion=arima_criterion,
                            method=arima_optimizer,  # default is lbfgs
                            trace=False,
                            error_action="ignore"))
    ])

    try:
        # Fit
        pipe.fit(df_ts['sales_quantity'])

        # Predict
        forecast = pipe.predict(prediction_length)

    except ValueError:
        raise ValueError(f'Crash during cutoff {cutoff}, on model {model_id}!')

    # Format
    df_forecast = pd.DataFrame()
    df_forecast['forecast'] = forecast
    df_forecast['forecast'] = df_forecast['forecast'].astype('float').round().astype(int).clip(lower=0)
    df_forecast['model_id'] = model_id
    df_forecast['cutoff'] = cutoff
    future_date_range = pd.date_range(start=week_id_to_date(cutoff), periods=prediction_length, freq='W')
    future_date_range_weeks = [date_to_week_id(w) for w in future_date_range]
    df_forecast['week_id'] = future_date_range_weeks
    df_forecast = df_forecast[['cutoff', 'model_id', 'week_id', 'forecast']]

    return df_forecast


def fit_predict_all_ts(df_predict, params, num_cpus):
    """
    Produces forecasts for each time series in the input dataframe.

    Args:
        df_predict (pd.DataFrame): DataFrame containing all the time series for which a forecast must be provided
        params (dict): Dictionary containing all the parameters required for the forecast

    Returns:
        df_forecast (pd.DataFrame): The output forecast DataFrame
    """

    # Drop useless params for fit-predict
    fit_predict_params = params.copy()
    fit_predict_params.pop('input_file_name')
    fit_predict_params.pop('s3_ouput_path')
    fit_predict_params.pop('context_length')

    print(f"Number of time series to forecast for cutoff {params['cutoff']}: {df_predict['model_id'].nunique()}")
    print(f"Launch of parallel forecasts over {num_cpus} cpus.")
    l_df_forecast = \
        Parallel(n_jobs=num_cpus,
                 verbose=1)(
                     delayed(_fit_predict_ts)(df_ts,
                                              **fit_predict_params) for _, df_ts in df_predict.groupby(['model_id']))

    df_forecast = pd.concat(l_df_forecast).sort_values(['model_id', 'week_id']).reset_index(drop=True)

    return df_forecast


def write_forecast_df_on_s3(df_forecast, params):
    """
    Produces forecasts for each time series in the input dataframe.

    Args:
        df_forecast (pd.DataFrame): The output forecast DataFrame
        params (dict): Dictionary containing all needed paths

    """
    bucket, dir_path = from_uri(params['s3_ouput_path'])
    file_path = f"{dir_path}{params['input_file_name']}.out"

    write_df_to_parquet_on_s3(df_forecast, bucket, file_path, verbose=True)
