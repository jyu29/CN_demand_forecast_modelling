import argparse
import boto3
import pandas as pd
import numpy as np
import re
import isoweek
from joblib import Parallel, delayed
import subprocess
import pickle
import os
from os import listdir
from os.path import isfile, join
import sys

from gluonts.model.prophet import ProphetPredictor

import utils as ut


def train_input_fn(train_file_path):

    with open(train_file_path, 'rb') as file:
        response = pickle.load(file)

    return response
    
    
def compute_wape(res):

    # This function is necessary for the hyperop part, to enable SageMaker's hyperopt module to choose the right parameters based on the WAPE
    active_sales = pd.read_parquet(os.environ["SM_DATA_DIR"] + '/active_sales')
    active_sales['date'] = pd.to_datetime(active_sales['date']) # Since the predictions' date column is a datetime, this step is necessary for a smooth merge right after

    res = pd.merge(res, active_sales, how="left")
    res["ae"] = np.abs(res["yhat"] - res["y"])
        
    cutoff_abs_error = res["ae"].sum()
    cutoff_target_sum = res["y"].sum()
    
    print(res.head())
    print("WAPE computed!", cutoff_abs_error, cutoff_target_sum)
    
    return cutoff_abs_error, cutoff_target_sum

    
def model_fn(cutoff_weeks, config, hyperparameters, n_jobs):
       
    def configure_seasonality(model):
        model.add_seasonality(
            name='yearly', 
            period=365.25, 
            fourier_order=hyperparameters['yearly_order'],
        )
        model.add_seasonality(
            name='quaterly',
            period=365.25/2, 
            fourier_order=hyperparameters['quaterly_order'], 
        )
        return model

    estimator = ProphetPredictor(
        freq=config.get_prediction_freq(),
        prediction_length=config.get_prediction_length(),
        prophet_params={'weekly_seasonality' : False,
                        'daily_seasonality' : False,
                        'yearly_seasonality' : False,
                        'n_changepoints' : hyperparameters['n_changepoints'],
                        'changepoint_range' : hyperparameters['changepoint_range'],
                        'changepoint_prior_scale' : hyperparameters['changepoint_prior_scale'],
                        'seasonality_prior_scale' : hyperparameters['seasonality_prior_scale']},
        init_model=configure_seasonality)
    
    def forecast_ts(ts):

        predictor = estimator.predict([ts])
        forecasts = list(predictor)
        
        week_id_range = ut.get_next_n_week(cutoff_week_id, config.get_horizon())
        
        res_ts = pd.DataFrame({
            'cutoff_week_id' : cutoff_week_id,
            'cutoff_date' : ut.week_id_to_date(cutoff_week_id),
            'week_id' : week_id_range,
            'date' : [ut.week_id_to_date(w) for w in week_id_range],
            'model' : np.repeat(ts['model'], config.get_prediction_length()),
            'yhat' : np.array([x.quantile(0.5).round().astype(int) \
                               for x in forecasts]).flatten()
        })
    
        return res_ts

    all_res = []
    
    for cutoff_week_id in cutoff_weeks:
        
        print("Forecasting cutoff "+ str(cutoff_week_id) + "...")
        
        train = train_input_fn(os.environ['SM_CHANNEL_TRAIN'] + '/gluonts_ds_cutoff_' + str(cutoff_week_id) + '.pkl')
                
        res = pd.concat(Parallel(n_jobs=n_jobs, verbose=1) \
                       (delayed(forecast_ts)(ts) for ts in train))
    
        res.loc[res['yhat'] < 0, 'yhat'] = 0
        
        ut.write_csv_S3(res, config.get_train_bucket_output(),
                        config.get_train_path_refined_data_output() + 'Facebook_Prophet_cutoff_' + str(cutoff_week_id) + '.csv')
        
        all_res.append(compute_wape(res))
     
    return all_res
    
    
def train_model_fn(cutoff_files_path, config, hyperparameters, n_jobs=-1, only_last=True):
                                    
    cutoff_files = [f for f in listdir(cutoff_files_path) if isfile(join(cutoff_files_path, f))]

    print(cutoff_files)

    cutoff_weeks = np.sort([int(re.findall('\d+', f)[0]) for f in cutoff_files if f.startswith('gluonts_ds_cutoff_')])

    print('Available cutoff weeks:', cutoff_weeks)

    if only_last:
        cutoff_weeks = np.array([np.max(cutoff_weeks)])
    
    print('Training cutoff(s):', cutoff_weeks)
        
    all_res = model_fn(cutoff_weeks, config, hyperparameters, n_jobs)
    
    l_cutoff_abs_error = [x[0] for x in all_res]
    l_cutoff_target_sum = [x[1] for x in all_res]
    
    l_cutoff_wape = np.array(l_cutoff_abs_error) / np.array(l_cutoff_target_sum)
    global_wape = np.sum(l_cutoff_abs_error) / np.sum(l_cutoff_target_sum)
    
    print("\n--------------------------------\n")
    print("cutoff_wape:", str(l_cutoff_wape))
    print("global_wape:", str(global_wape))
    print("\n--------------------------------\n")