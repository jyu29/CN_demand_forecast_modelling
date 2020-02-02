import os
import pandas as pd
import numpy as np
import re
from gluonts.dataset.common import ListDataset
from gluonts.transform import FieldName
import pickle

import utils as ut



def write_train_input_fn(pkl_object, train_file_path):
    with open(train_file_path, 'wb') as file:
        pickle.dump(pkl_object, file)
        
        
def format_cutoff_train_data(train_dir, config, only_last=True):

    #cutoff_files = ut.get_s3_subdirectories(config.get_train_bucket_input(), 
    #                                        config.get_train_path_refined_data_input())
    
    cutoff_files = next(os.walk(train_dir))[1]
    
    # Remove prefixes from file paths (to make sure the scope name's digits dont get parsed), and exctract week number
    #cutoff_weeks = np.array([int(re.findall('\d+', f.split('/')[-2])[0]) for f in cutoff_files])
    cutoff_weeks = np.array([int(re.findall('\d+', f)[0]) for f in cutoff_files])
    
    if only_last:
        cutoff_weeks = np.array([np.max(cutoff_weeks)])
    else:
        ut.delete_S3(config.get_train_bucket_input(), config.get_train_path_refined_data_intermediate())

    for cutoff_week in cutoff_weeks:
    
        print('Generating GluonTS dataset for cutoff ' + str(cutoff_week) + '...')
    
        train_data_cutoff = pd.read_parquet(train_dir +\
                                           '/train_data_cutoff_' +\
                                           str(cutoff_week) + '/')
    
        train_data_cutoff = train_data_cutoff.sort_values('week_id')
        train_data_cutoff = train_data_cutoff\
            .groupby('model', as_index=False)\
            .agg({'y' : list, 'date' : min})
    
        gluonts_ds = ListDataset([{'model' : row['model'],
                                   FieldName.TARGET: row['y'],
                                   FieldName.START: row['date']
                                  } for _, row in train_data_cutoff.iterrows()],
                                 freq=config.get_prediction_freq())
        
        
        write_train_input_fn(gluonts_ds, train_dir + '/gluonts_ds_cutoff_' +\
                           str(cutoff_week) + '.pkl')
        
        ut.write_pickle_S3(gluonts_ds,
                           config.get_train_bucket_input(),
                           config.get_train_path_refined_data_intermediate() + 'gluonts_ds_cutoff_' +\
                           str(cutoff_week) + '.pkl')