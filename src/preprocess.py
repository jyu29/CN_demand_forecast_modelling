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
    
    cutoff_files = next(os.walk(train_dir))[1]
    
    # Extract week_number # todo : check if week = current week
    cutoff_weeks = np.array([int(re.findall('\d+', f)[0]) for f in cutoff_files if f.startswith('train_data_cutoff_') and f[-1].isdigit()])

    print("Cutoff weeks:", cutoff_weeks)

    if only_last:
        cutoff_weeks = np.array([np.max(cutoff_weeks)])

    print(only_last)
    print(cutoff_weeks)

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