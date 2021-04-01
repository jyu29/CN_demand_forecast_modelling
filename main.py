import argparse
import json
import os

import pandas as pd

import src.data_handler as dh
import src.sagemaker_utils as su
import src.utils as ut


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging_lvl', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        default="INFO", help="Level for logger")
    parser.add_argument('--environment', choices=['dev', 'prod', 'dev_old'], default="dev",
                        help="'dev' or 'prod', to set the right configurations")
    parser.add_argument('--list_cutoff', default=str('today'), help="List of cutoffs in format YYYYWW between brackets or 'today'")
    parser.add_argument('--run_name', help="Run Name for file hierarchy purpose")
    args = parser.parse_args()
    su.logger.setLevel(args.logging_lvl)
    dh.logger.setLevel(args.logging_lvl)
    
    # Defining variables
    environment = args.environment
    if args.list_cutoff == 'today':
        list_cutoff = [ut.get_current_week()]
    else:
        list_cutoff = json.loads(args.list_cutoff)

    for cutoff in list_cutoff:
        assert type(cutoff) == int

    assert type(args.run_name) == str
    run_name = args.run_name

    # Constants
    REFINED_BUCKET = ut.import_raw_config(environment)['buckets']['refined_data_specific']
    REFINED_DATA_SPECIFIC_PATH = ut.to_uri(ut.import_raw_config(environment)['buckets']['refined_data_specific'],
                                           ut.import_raw_config(environment)['paths']['refined_specific_path']
                                           )
    ALGORITHM = ut.import_raw_config(environment)['modeling_parameters']['algorithm']
    MODEL_WEEK_SALES_PATH = 'global/model_week_sales'
    MODEL_WEEK_TREE_PATH = 'global/model_week_tree'
    MODEL_WEEK_MRP_PATH = 'global/model_week_mrp'
    IMPUTED_SALES_LOCKDOWN_1_PATH = 'global/imputed_sales_lockdown_1'

    # Data loading
    df_model_week_sales = ut.read_multipart_parquet_s3(REFINED_BUCKET, MODEL_WEEK_SALES_PATH)
    df_model_week_tree = ut.read_multipart_parquet_s3(REFINED_BUCKET, MODEL_WEEK_TREE_PATH)
    df_model_week_mrp = ut.read_multipart_parquet_s3(REFINED_BUCKET, MODEL_WEEK_MRP_PATH)
    df_store_openings = pd.read_csv('data/store_openings.csv', sep=";")
    df_holidays = pd.read_csv('data/holidays.csv', sep=";")
    df_imputed_sales_lockdown_1 = ut.read_multipart_parquet_s3(REFINED_BUCKET, IMPUTED_SALES_LOCKDOWN_1_PATH)

    # Generate empty df_jobs
    df_jobs = su.generate_df_jobs(list_cutoff=list_cutoff,
                                  run_name=run_name,
                                  algorithm=ALGORITHM,
                                  refined_data_specific_path=REFINED_DATA_SPECIFIC_PATH
                                  )

    for cutoff in list_cutoff:
        # Parameters init
        TRAIN_PATH = df_jobs[df_jobs['cutoff'] == cutoff].loc[:, 'train_path'].values[0]
        PREDICT_PATH = df_jobs[df_jobs['cutoff'] == cutoff].loc[:, 'predict_path'].values[0]

        refining_params = dh.import_refining_config(environment=environment,
                                                    cutoff=cutoff,
                                                    run_name=run_name,
                                                    train_path=TRAIN_PATH,
                                                    predict_path=PREDICT_PATH
                                                    )


        # Data/Features init
        base_data = {'model_week_sales': df_model_week_sales,
                     'model_week_tree': df_model_week_tree,
                     'model_week_mrp': df_model_week_mrp,
                     'imputed_sales_lockdown_1': df_imputed_sales_lockdown_1
                     }

        df_static_tree = df_model_week_tree[df_model_week_tree['week_id'] == cutoff].copy()
        static_features = {'model_identifier': pd.DataFrame({'model_id': df_static_tree['model_id'],
                                                             'model_identifier': df_static_tree['model_id']}),
                           'family_id': df_static_tree[['model_id', 'family_id']],
                           'sub_department_id': df_static_tree[['model_id', 'sub_department_id']],
                           'department_id': df_static_tree[['model_id', 'department_id']],
                           'product_nature_id': df_static_tree[['model_id', 'product_nature_id']],
                           'univers_id': df_static_tree[['model_id', 'univers_id']],
                           }

        global_dynamic_features = {'store_openings': {'dataset': df_store_openings,
                                                      'projection': 'ffill'},
                                   'holidays': {'dataset': df_holidays,
                                                'projection': 'as_provided'}
                                   }

        specific_dynamic_features = None

        refining_handler = dh.DataHandler(base_data=base_data,
                                          static_features=static_features,
                                          global_dynamic_features=global_dynamic_features,
                                          specific_dynamic_features=specific_dynamic_features,
                                          **refining_params
                                          )

        refining_handler.execute_data_refining_specific()

    sagemaker_params = su.import_sagemaker_params(environment=environment)

    modeling_handler = su.SagemakerHandler(run_name=run_name,
                                           df_jobs=df_jobs,
                                           **sagemaker_params)

    modeling_handler.launch_training_jobs()

    modeling_handler.launch_transform_jobs()
