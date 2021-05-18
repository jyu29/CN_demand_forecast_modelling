import os
import logging

import pandas as pd

import src.data_handler as dh
import src.sagemaker_utils as su
import src.model_stacking as ms
import src.utils as ut


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    
    # Modeling arguments handling
    ENVIRONMENT = os.environ['run_env']
    LIST_CUTOFF = os.environ['list_cutoff']
    RUN_NAME = os.environ['run_name']

    ut.check_environment(ENVIRONMENT)
    list_cutoff = ut.check_list_cutoff(LIST_CUTOFF)
    ut.check_run_name(RUN_NAME)

    # Logging level
    try:
        LOGGING_LVL = os.environ['logging_lvl']
        assert LOGGING_LVL in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], 'Wrong logging level'
    except KeyError:
        LOGGING_LVL = 'INFO'
        logger.info("Logging level set to INFO")

    for module in [dh, su, ms]:
        module.logger.setLevel(LOGGING_LVL)

    # Constants
    main_params = ut.import_modeling_parameters(ENVIRONMENT)
    REFINED_DATA_GLOBAL_BUCKET = main_params['refined_data_global_bucket']
    REFINED_DATA_SPECIFIC_BUCKET = main_params['refined_data_specific_bucket']
    REFINED_DATA_GLOBAL_PATH = main_params['refined_global_path']
    REFINED_DATA_SPECIFIC_PATH = main_params['refined_specific_path']
    REFINED_DATA_SPECIFIC_URI = ut.to_uri(REFINED_DATA_SPECIFIC_BUCKET, REFINED_DATA_SPECIFIC_PATH)
    MODEL_WEEK_SALES_PATH = f"{REFINED_DATA_GLOBAL_PATH}model_week_sales"
    MODEL_WEEK_TREE_PATH = f"{REFINED_DATA_GLOBAL_PATH}model_week_tree"
    MODEL_WEEK_MRP_PATH = f"{REFINED_DATA_GLOBAL_PATH}model_week_mrp"
    IMPUTED_SALES_LOCKDOWN_1_PATH = f"{REFINED_DATA_GLOBAL_PATH}imputed_sales_lockdown_1.parquet"
    LIST_ALGORITHM = list(main_params['algorithm'])
    DEEPAR_ARIMA_STACKING = main_params['deepar_arima_stacking']

    # Data loading
    df_model_week_sales = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, MODEL_WEEK_SALES_PATH)
    df_model_week_tree = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, MODEL_WEEK_TREE_PATH)
    df_model_week_mrp = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, MODEL_WEEK_MRP_PATH)
    df_imputed_sales_lockdown_1 = ut.read_multipart_parquet_s3(REFINED_DATA_GLOBAL_BUCKET, IMPUTED_SALES_LOCKDOWN_1_PATH)

    # Initialize df_jobs
    df_jobs = su.generate_df_jobs(list_cutoff=list_cutoff,
                                  run_name=RUN_NAME,
                                  list_algorithm=LIST_ALGORITHM,
                                  refined_data_specific_path=REFINED_DATA_SPECIFIC_URI
                                  )

    # Generate modeling specific data
    for _, job in df_jobs.iterrows():

        # Parameters init
        algorithm = job['algorithm']
        cutoff = job['cutoff']
        train_path = job['train_path']
        predict_path = job['predict_path']

        refining_params = dh.import_refining_config(environment=ENVIRONMENT,
                                                    algorithm=algorithm,
                                                    cutoff=cutoff,
                                                    train_path=train_path,
                                                    predict_path=predict_path
                                                    )

        # Data/Features init
        base_data = {
            'model_week_sales': df_model_week_sales,
            'model_week_tree': df_model_week_tree,
            'model_week_mrp': df_model_week_mrp,
            'imputed_sales_lockdown_1': df_imputed_sales_lockdown_1
        }

        df_static_tree = df_model_week_tree[df_model_week_tree['week_id'] == cutoff].copy()

        static_features = {
            'family_id': df_static_tree[['model_id', 'family_id']],
            'sub_department_id': df_static_tree[['model_id', 'sub_department_id']],
            'department_id': df_static_tree[['model_id', 'department_id']],
            'univers_id': df_static_tree[['model_id', 'univers_id']],
            'product_nature_id': df_static_tree[['model_id', 'product_nature_id']]
        }

        global_dynamic_features = None

        specific_dynamic_features = None

        # Execute data refining
        refining_handler = dh.DataHandler(base_data=base_data,
                                          static_features=static_features,
                                          global_dynamic_features=global_dynamic_features,
                                          specific_dynamic_features=specific_dynamic_features,
                                          **refining_params
                                          )

        refining_handler.execute_data_refining_specific()


    # Launch Fit & Transform
    for algorithm in LIST_ALGORITHM:

        df_jobs_algo = df_jobs[df_jobs['algorithm'] == algorithm].copy()

        sagemaker_params = su.import_sagemaker_params(environment=ENVIRONMENT, algorithm=algorithm)

        modeling_handler = su.SagemakerHandler(df_jobs=df_jobs_algo, **sagemaker_params)

        modeling_handler.launch_training_jobs()

        if algorithm == 'deepar':
            modeling_handler.launch_transform_jobs()

    # Calculate model stacking
    if DEEPAR_ARIMA_STACKING:
        ms.calculate_deepar_arima_stacking(df_jobs)
