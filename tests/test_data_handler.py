import os
import pandas as pd
import pandas.api.types as ptypes
import pytest
from pytest import mark

from src.data_handler import DataHandler, import_refining_config


DATA_PATH = os.path.join('tests', 'data')
ENVIRONMENT = 'testing'
CUTOFF = 202001
RUN_NAME = 'testing_run_name'
TRAIN_PATH = 's3://fcst-refined-demand-forecast-dev/specific/testing/deepAR/testrun-20201/input/train_202001.json'
PREDICT_PATH = 's3://fcst-refined-demand-forecast-dev/specific/testing/deepAR/testrun-202021/input/predict_202001.json'

df_model_week_sales = pd.read_csv(os.path.join(DATA_PATH, "model_week_sales.csv"), sep=';', parse_dates=['date'])
df_model_week_tree = pd.read_csv(os.path.join(DATA_PATH, "model_week_tree.csv"), sep=';')
df_model_week_mrp = pd.read_csv(os.path.join(DATA_PATH, "model_week_mrp.csv"), sep=';')
df_store_openings = pd.read_csv(os.path.join(DATA_PATH, "store_openings.csv"), sep=';')
df_holidays = pd.read_csv(os.path.join(DATA_PATH, "holidays.csv"), sep=';')
df_imputed_sales_lockdown_1 = pd.read_csv(os.path.join(DATA_PATH, "model_week_imputed_lockdown_1.csv"), sep=';', parse_dates=['date'])

base_data = {'model_week_sales': df_model_week_sales,
             'model_week_tree': df_model_week_tree,
             'model_week_mrp': df_model_week_mrp,
             'imputed_sales_lockdown_1': df_imputed_sales_lockdown_1
             }

df_static_tree = df_model_week_tree[df_model_week_tree['week_id'] == CUTOFF].copy()

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

refining_params = import_refining_config(environment=ENVIRONMENT,
                                         cutoff=CUTOFF,
                                         run_name=RUN_NAME,
                                         train_path=TRAIN_PATH,
                                         predict_path=PREDICT_PATH
                                         )

data_handler = DataHandler(base_data=base_data,
                           static_features=static_features,
                           global_dynamic_features=global_dynamic_features,
                           specific_dynamic_features=specific_dynamic_features,
                           **refining_params
                           )


@mark.data_handler
class ImportRefiningConfigTests():
    def test_nominal(self):
        expected_config = {'cutoff': 202001,
                           'patch_first_lockdown': True,
                           'rec_cold_start': True,
                           'rec_length': 104,
                           'rec_cold_start_group': ['family_id'],
                           'prediction_length': 52,
                           'refined_global_bucket': 'fcst-refined-demand-forecast-dev',
                           'refined_specific_bucket': 'fcst-refined-demand-forecast-dev',
                           'output_paths': {'train_path': TRAIN_PATH, 'predict_path': PREDICT_PATH}
                           }

        config = import_refining_config(environment=ENVIRONMENT,
                                        cutoff=CUTOFF,
                                        run_name=RUN_NAME,
                                        train_path=TRAIN_PATH,
                                        predict_path=PREDICT_PATH)

        try:
            assert expected_config == config
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_wrong_type(self):
        environment = 202001
        cutoff = 'test'
        run_name = {'foo': 'bar'}
        train_path = ['wrong_path', 1234]

        with pytest.raises(AssertionError):
            import_refining_config(environment=environment,
                                   cutoff=CUTOFF,
                                   run_name=RUN_NAME,
                                   train_path=TRAIN_PATH,
                                   predict_path=PREDICT_PATH)

        with pytest.raises(AssertionError):
            import_refining_config(environment=ENVIRONMENT,
                                   cutoff=cutoff,
                                   run_name=RUN_NAME,
                                   train_path=TRAIN_PATH,
                                   predict_path=PREDICT_PATH)

        with pytest.raises(AssertionError):
            import_refining_config(environment=ENVIRONMENT,
                                   cutoff=CUTOFF,
                                   run_name=run_name,
                                   train_path=TRAIN_PATH,
                                   predict_path=PREDICT_PATH)

        with pytest.raises(AssertionError):
            import_refining_config(environment=ENVIRONMENT,
                                   cutoff=CUTOFF,
                                   run_name=RUN_NAME,
                                   train_path=train_path,
                                   predict_path=PREDICT_PATH)


class DataHandlerImportBaseDataTests:
    def test_date_parsing(self):
        data_handler.import_base_data()
        
        try:
            assert ptypes.is_datetime64_ns_dtype(data_handler.base_data['model_week_sales']['date'])
        except AssertionError:
            pytest.fail("Test failed on nominal case")
