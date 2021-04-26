import json
import os
import logging

import pandas as pd

import src.data_handler as dh
import src.sagemaker_utils as su
import src.utils as ut


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--logging_lvl',
    #                     choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
    #                     default="INFO",
    #                     help="Level for logger"
    #                     )
    # parser.add_argument('--environment',
    #                     choices=['dev', 'prod', 'dev_old'],
    #                     default="dev",
    #                     help="'dev' or 'prod', to set the right configurations"
    #                     )
    # parser.add_argument('--list_cutoff',
    #                     default=str('today'),
    #                     help="List of cutoffs in format YYYYWW between brackets or 'today'"
    #                     )
    # parser.add_argument('--run_name', help="Run Name for file hierarchy purpose")

    # Modeling arguments handling
    ENVIRONMENT = os.environ['environment']
    LIST_CUTOFF = os.environ['list_cutoff']
    RUN_NAME = os.environ['run_name']

    ut.check_environment(ENVIRONMENT)
    list_cutoff = ut.check_list_cutoff(LIST_CUTOFF)
    ut.check_run_name(RUN_NAME)

    try:
        LOGGING_LVL = os.environ['logging_lvl']
        assert LOGGING_LVL in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], 'Wrong logging level'
    except KeyError:
        LOGGING_LVL = 'INFO'
        logger.info("Logging level set to INFO")

    for module in [ut, su, dh]:
        module.logger.setLevel(LOGGING_LVL)

    # Constants
    main_params = ut.import_modeling_parameters(ENVIRONMENT)
    REFINED_DATA_GLOBAL_BUCKET = main_params['refined_data_global_bucket']
    REFINED_DATA_SPECIFIC_BUCKET = main_params['refined_data_global_bucket']

    REFINED_DATA_GLOBAL_PATH = main_params['refined_global_path']
    REFINED_DATA_SPECIFIC_PATH = main_params['refined_specific_path']
    REFINED_DATA_SPECIFIC_URI = ut.to_uri(REFINED_DATA_SPECIFIC_BUCKET, REFINED_DATA_SPECIFIC_PATH)
    ALGORITHM = main_params['algorithm']
    MODEL_WEEK_SALES_PATH = f"{REFINED_DATA_GLOBAL_PATH}model_week_sales"
    MODEL_WEEK_TREE_PATH = f"{REFINED_DATA_GLOBAL_PATH}model_week_tree"
    MODEL_WEEK_MRP_PATH = f"{REFINED_DATA_GLOBAL_PATH}model_week_mrp"
    IMPUTED_SALES_LOCKDOWN_1_PATH = f"{REFINED_DATA_GLOBAL_PATH}imputed_sales_lockdown_1.parquet"
    STORE_OPENINGS_PATH = f"{REFINED_DATA_GLOBAL_PATH}store_openings/store_openings.parquet"
    HOLIDAYS_PATH = f"{REFINED_DATA_GLOBAL_PATH}holidays/holidays.parquet"

    # Data loading
    df_model_week_sales = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, MODEL_WEEK_SALES_PATH)
    df_model_week_tree = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, MODEL_WEEK_TREE_PATH)
    df_model_week_mrp = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, MODEL_WEEK_MRP_PATH)
    df_imputed_sales_lockdown_1 = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET,
                                                               IMPUTED_SALES_LOCKDOWN_1_PATH)
    df_store_openings = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, STORE_OPENINGS_PATH)
    df_holidays = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, HOLIDAYS_PATH)

    # Generate empty df_jobs
    df_jobs = su.generate_df_jobs(list_cutoff=list_cutoff,
                                  run_name=RUN_NAME,
                                  algorithm=ALGORITHM,
                                  refined_data_specific_path=REFINED_DATA_SPECIFIC_URI
                                  )

    for cutoff in list_cutoff:
        # Parameters init
        TRAIN_PATH = df_jobs[df_jobs['cutoff'] == cutoff].loc[:, 'train_path'].values[0]
        PREDICT_PATH = df_jobs[df_jobs['cutoff'] == cutoff].loc[:, 'predict_path'].values[0]

        refining_params = dh.import_refining_config(environment=ENVIRONMENT,
                                                    cutoff=cutoff,
                                                    run_name=ENVIRONMENT,
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

    sagemaker_params = su.import_sagemaker_params(environment=ENVIRONMENT)

    modeling_handler = su.SagemakerHandler(run_name=RUN_NAME,
                                           df_jobs=df_jobs,
                                           **sagemaker_params)

    modeling_handler.launch_training_jobs()

    modeling_handler.launch_transform_jobs()
