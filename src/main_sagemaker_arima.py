import argparse
import os

import pandas as pd

from arima_functions import fit_predict_all_ts, write_forecast_df_on_s3

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # Config params
    parser.add_argument("--input_file_name", type=str)
    parser.add_argument("--s3_ouput_path", type=str)
    parser.add_argument("--cutoff", type=str)
    
    # Model params
    parser.add_argument("--prediction_length", type=int, default=104)
    parser.add_argument("--context_length", type=int, default=156)
    parser.add_argument("--fourier_seasonal_period", type=int, default=52)
    parser.add_argument("--fourier_order", type=int, default=None)
    parser.add_argument("--arima_differencing_order", type=int, default=None)
    parser.add_argument("--arima_criterion", type=str, default='aic')
    parser.add_argument("--arima_optimizer", type=int, default='lbfgs')
    
    params = parser.parse_args()

    # Load input df
    df_predict = pd.read_parquet(os.path.join(os.environ["SM_CHANNEL_TRAINING"], params['input_file_name']))
    
    # Get forecasts
    df_forecast = fit_predict_all_ts(df_predict, params, num_cpus=os.environ['SM_NUM_CPUS'])
    
    # Write forecasts on s3
    write_forecast_df_on_s3(df_forecast, params)