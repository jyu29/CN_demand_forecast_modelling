import pandas as pd
import numpy as np
import re
from gluonts.dataset.common import ListDataset
from gluonts.transform import FieldName

import utils as ut


def format_cutoff_train_data(conf, only_last=True):

    cutoff_files = ut.get_files_list(config.bucket, 
                                     config.s3_path_refined_data + 'cutoff_data/')
    cutoff_weeks = np.array([int(re.findall('\d+', f)[0]) for f in cutoff_files])
    
    if only_last:
        cutoff_weeks = np.array([np.max(cutoff_weeks)])
    else:
        ut.delete_S3(cf.bucket, cf.s3_path_refined_data + 'gluonts_data/')

    for cutoff_week in cutoff_weeks:
    
        print('Generating GluonTS dataset for cutoff ' + str(cutoff_week) + '...')
    
        train_data_cutoff = ut.read_csv_S3(conf.bucket,
                                           conf.s3_path_refined_data +\
                                           'cutoff_data/train_data_cutoff_' +\
                                           str(cutoff_week) + '.csv',
                                           parse_dates=['date'])
    
        train_data_cutoff = train_data_cutoff.sort_values('week_id')
        train_data_cutoff = train_data_cutoff\
            .groupby('model', as_index=False)\
            .agg({'y' : list, 'date' : min})
    
        gluonts_ds = ListDataset([{'model' : row['model'],
                                   FieldName.TARGET: row['y'],
                                   FieldName.START: row['date']
                                  } for _, row in train_data_cutoff.iterrows()],
                                 freq=cf.prediction_freq)
    
        ut.write_pickle_S3(gluonts_ds,
                           cf.bucket,
                           cf.s3_path_refined_data + 'gluonts_data/gluonts_ds_cutoff_' +\
                           str(cutoff_week) + '.pkl')