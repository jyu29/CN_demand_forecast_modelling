import os
import pandas as pd

from sagemaker_arima_functions import (import_parameters, fit_predict_all_ts, write_forecast_df_on_s3)

if __name__ == "__main__":
    
    # Get parameters
    config_params, hyperparameters = import_parameters(os.environ['HPS_FILE_PATH'])
    
    # Load input df
    df_predict = pd.read_parquet(os.path.join(os.environ['INPUT_DATA_DIR'], config_params['input_file_name']))

    # Calculate forecasts
    df_forecast = fit_predict_all_ts(df_predict, hyperparameters)
    
    # Write forecasts on s3
    write_forecast_df_on_s3(df_forecast, config_params['s3_output_path'], config_params['input_file_name'])