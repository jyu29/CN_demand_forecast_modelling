import argparse

import pandas as pd

import src.utils as ut
import src.data_handler as dh
import src.sagemaker_utils as su

parser = argparse.ArgumentParser()
parser.add_argument('--logging_lvl', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                    default="INFO", help="Level for logger")
args = parser.parse_args()
su.logger.setLevel(args.logging_lvl)
dh.logger.setLevel(args.logging_lvl)

list_cutoff = [202105]
run_name = 'testrun'
environment = 'test'

# Data loading
df_model_week_sales = pd.read_parquet('~/Downloads/modeling_data/model_week_sales/')
df_model_week_tree = pd.read_parquet('~/Downloads/modeling_data/model_week_tree/')
df_model_week_mrp = pd.read_parquet('~/Downloads/modeling_data/model_week_mrp/')
df_store_openings = pd.read_csv('~/Downloads/modeling_data/store_openings/store_openings.csv', sep=";")
df_holidays = pd.read_csv('~/Downloads/modeling_data/holidays/holidays.csv', sep=";")
df_test_feat = pd.read_csv('~/Downloads/modeling_data/test_dynamic_specific_feat/test_feat.csv', sep=";")

# Generate empty df_jobs
refined_data_specific_path = ut.to_uri(ut.import_raw_config(environment)['buckets']['refined_data_specific'],
                                       ut.import_raw_config(environment)['paths']['refined_specific_path']
                                       )
algorithm = ut.import_raw_config(environment)['modeling_parameters']['algorithm']
df_jobs = su.generate_df_jobs(list_cutoff=list_cutoff,
                              run_name=run_name,
                              algorithm=algorithm,
                              refined_data_specific_path=refined_data_specific_path
                              )

if __name__ == "__main__":
    for cutoff in list_cutoff:
        # Data/Features init
        base_data = {'model_week_sales': df_model_week_sales,
                     'model_week_tree': df_model_week_tree,
                     'model_week_mrp': df_model_week_mrp
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
        
        train_path = df_jobs[df_jobs['cutoff'] == cutoff].loc[:, 'train_path'].values[0]
        predict_path = df_jobs[df_jobs['cutoff'] == cutoff].loc[:, 'predict_path'].values[0]

        refining_params = ut.import_refining_config(environment=environment,
                                                    cutoff=cutoff,
                                                    run_name=run_name,
                                                    train_path=train_path,
                                                    predict_path=predict_path
                                                    )

        refining_handler = dh.data_handler(base_data=base_data,
                                           static_features=static_features,
                                           global_dynamic_features=global_dynamic_features,
                                           specific_dynamic_features=specific_dynamic_features,
                                           **refining_params
                                           )

        refining_handler.execute_data_refining_specific()

    sagemaker_params = ut.import_sagemaker_params(environment=environment)

    # modeling_handler = su.SagemakerHandler(run_name=run_name,
    #                                        df_jobs=df_jobs,
    #                                        **sagemaker_params)

    # modeling_handler.launch_training_jobs()

    # modeling_handler.launch_transform_jobs()
