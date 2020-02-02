import argparse
import boto3
import pandas as pd
import numpy as np
import re
import isoweek
#from copy import deepcopy
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

    active_sales = pd.read_csv('/opt/ml/input/data/active_sales.csv', sep='|', parse_dates=['date'])
                              
    res = pd.merge(res, active_sales, how="left")
    res["ae"] = np.abs(res["yhat"] - res["y"])
        
    cutoff_abs_error = res["ae"].sum()
    cutoff_target_sum = res["y"].sum()
    
    print(res.head())
    print("WAPE computed!", cutoff_abs_error, cutoff_target_sum)
    
    return cutoff_abs_error, cutoff_target_sum

    
def model_fn(cutoff_week_id, config, hyperparameters):

    train = train_input_fn(os.environ['SM_CHANNEL_TRAIN'] + '/gluonts_ds_cutoff_' + str(cutoff_week_id) + '.pkl')
        
    nb_ts = len(train)
    
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
    
    predictor = estimator.predict(train)
    forecasts = list(predictor)
    
    week_id_range = ut.get_next_n_week(cutoff_week_id, config.get_horizon())
    
    res = pd.DataFrame(
        {'cutoff_week_id' : cutoff_week_id,
         'cutoff_date' : ut.week_id_to_date(cutoff_week_id),
         'week_id' : week_id_range * nb_ts,
         'date' : [ut.week_id_to_date(w) for w in week_id_range] * nb_ts,
         'model' : np.array([np.repeat(x['model'], config.get_prediction_length()) for x in train]).flatten(),
         'yhat' : np.array([x.quantile(0.5).round().astype(int) for x in forecasts]).flatten()})
    
    res.loc[res['yhat'] < 0, 'yhat'] = 0
    
    #res.to_csv(model_dir + '/Facebook_Prophet_cutoff_' + str(cutoff_week_id) + '.csv')
    ut.write_csv_S3(res, config.get_train_bucket_output(),
                    config.get_train_path_refined_data_output()+'Facebook_Prophet_cutoff_' + str(cutoff_week_id) + '.csv')
    
    return compute_wape(res)
    
    
def train_model_fn(cutoff_files_path, config, hyperparameters, max_jobs=-1, only_last=True):
                                    
    cutoff_files = [f for f in listdir(cutoff_files_path) if isfile(join(cutoff_files_path, f))]

    cutoff_weeks = np.array([int(re.findall('\d+', f)[0]) for f in cutoff_files])
    
    if only_last:
        cutoff_weeks = np.array([np.min(cutoff_weeks)]) #max

    if max_jobs <= 0:
        max_jobs = len(cutoff_weeks)
        
    all_res = Parallel(n_jobs=max_jobs, verbose=1)\
            (delayed(model_fn)(cutoff_week_id, config, hyperparameters) for cutoff_week_id in cutoff_weeks)
    
    l_cutoff_abs_error = [x[0] for x in all_res]
    l_cutoff_target_sum = [x[1] for x in all_res]
    
    l_cutoff_wape = np.array(l_cutoff_abs_error) / np.array(l_cutoff_target_sum)
    global_wape = np.sum(l_cutoff_abs_error) / np.sum(l_cutoff_target_sum)
    
    print("\n--------------------------------\n")
    print("cutoff_wape:", str(l_cutoff_wape))
    print("global_wape:", str(global_wape))
    print("\n--------------------------------\n")