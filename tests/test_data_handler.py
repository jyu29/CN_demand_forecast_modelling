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
REFINED_TARGET_PATH = os.path.join(DATA_PATH, "refining_target.csv")
REFINED_STATIC_PATH = os.path.join(DATA_PATH, "refining_static_data.csv")
REFINED_DYNAMIC_PATH = os.path.join(DATA_PATH, "refining_dynamic_data.csv")

df_model_week_sales = pd.read_csv(os.path.join(DATA_PATH, "model_week_sales.csv"), sep=';', parse_dates=['date'])
df_model_week_tree = pd.read_csv(os.path.join(DATA_PATH, "model_week_tree.csv"), sep=';')
df_model_week_mrp = pd.read_csv(os.path.join(DATA_PATH, "model_week_mrp.csv"), sep=';')
df_store_openings = pd.read_csv(os.path.join(DATA_PATH, "store_openings.csv"), sep=';')
df_holidays = pd.read_csv(os.path.join(DATA_PATH, "holidays.csv"), sep=';')
df_imputed_sales_lockdown_1 = pd.read_csv(os.path.join(DATA_PATH, "model_week_imputed_lockdown_1.csv"),
                                          sep=';',
                                          parse_dates=['date']
                                          )

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


class DataHandlerRefiningSpecificTests:
    def test_nominal(self):
        data_handler.import_base_data()

        expected_target = pd.read_csv(REFINED_TARGET_PATH, sep=';')
        expected_static_data = pd.read_csv(REFINED_STATIC_PATH, sep=';')
        expected_dynamic_data = pd.read_csv(REFINED_DYNAMIC_PATH, sep=';')

        target, static_data, dynamic_data = data_handler.refining_specific()

        try:
            assert target.reset_index(drop=True).equals(expected_target.reset_index(drop=True))
            assert static_data.reset_index(drop=True).equals(expected_static_data.reset_index(drop=True))
            assert dynamic_data.reset_index(drop=True).equals(expected_dynamic_data.reset_index(drop=True))
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_isrec_no_feat(self):
        data_handler = DataHandler(base_data=base_data,
                                   static_features=None,
                                   global_dynamic_features=None,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        expected_target = pd.read_csv(REFINED_TARGET_PATH, sep=';')
        expected_dynamic_data = pd.read_csv(REFINED_DYNAMIC_PATH, sep=';')[['week_id', 'model_id', 'is_rec']]

        target, static_data, dynamic_data = data_handler.refining_specific()

        try:
            assert target.reset_index(drop=True).equals(expected_target.reset_index(drop=True))
            assert static_data is None
            assert dynamic_data.reset_index(drop=True).equals(expected_dynamic_data.reset_index(drop=True))
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_isrec_only_specific_dyn_feat(self):
        pass

    def test_nominal_isrec_only_global_dyn_feat(self):
        data_handler = DataHandler(base_data=base_data,
                                   static_features=None,
                                   global_dynamic_features=global_dynamic_features,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        expected_target = pd.read_csv(REFINED_TARGET_PATH, sep=';')
        expected_dynamic_data = pd.read_csv(REFINED_DYNAMIC_PATH, sep=';')

        target, static_data, dynamic_data = data_handler.refining_specific()

        try:
            assert target.reset_index(drop=True).equals(expected_target.reset_index(drop=True))
            assert static_data is None
            assert dynamic_data.reset_index(drop=True).equals(expected_dynamic_data.reset_index(drop=True))
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_isrec_only_static_feat(self):
        data_handler = DataHandler(base_data=base_data,
                                   static_features=static_features,
                                   global_dynamic_features=None,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        expected_target = pd.read_csv(REFINED_TARGET_PATH, sep=';')
        expected_static_data = pd.read_csv(REFINED_STATIC_PATH, sep=';')
        expected_dynamic_data = pd.read_csv(REFINED_DYNAMIC_PATH, sep=';')[['week_id', 'model_id', 'is_rec']]

        target, static_data, dynamic_data = data_handler.refining_specific()

        try:
            assert target.reset_index(drop=True).equals(expected_target.reset_index(drop=True))
            assert static_data.reset_index(drop=True).equals(expected_static_data.reset_index(drop=True))
            assert dynamic_data.reset_index(drop=True).equals(expected_dynamic_data.reset_index(drop=True))
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_norec_no_feat(self):
        refining_params['rec_cold_start'] = False

        data_handler = DataHandler(base_data=base_data,
                                   static_features=None,
                                   global_dynamic_features=None,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        expected_target = pd.read_csv(REFINED_TARGET_PATH, sep=';')

        target, static_data, dynamic_data = data_handler.refining_specific()

        try:
            assert target.reset_index(drop=True).equals(expected_target.reset_index(drop=True))
            assert static_data is None
            assert dynamic_data is None
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_norec_only_specific_dyn_feat(self):
        refining_params['rec_cold_start'] = False

        pass

    def test_nominal_norec_only_global_dyn_feat(self):
        refining_params['rec_cold_start'] = False

        data_handler = DataHandler(base_data=base_data,
                                   static_features=None,
                                   global_dynamic_features=global_dynamic_features,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        expected_target = pd.read_csv(REFINED_TARGET_PATH, sep=';')
        expected_dynamic_data = pd.read_csv(REFINED_DYNAMIC_PATH, sep=';')[['week_id',
                                                                            'model_id',
                                                                            'perc_store_open',
                                                                            'holidays'
                                                                            ]]

        target, static_data, dynamic_data = data_handler.refining_specific()

        test_dynamic_data = dynamic_data.merge(expected_dynamic_data, on=['week_id', 'model_id'])

        try:
            assert target.reset_index(drop=True).equals(expected_target.reset_index(drop=True))
            assert static_data is None
            assert (test_dynamic_data['perc_store_open_x'] == test_dynamic_data['perc_store_open_y']).all()
            assert (test_dynamic_data['holidays_x'] == test_dynamic_data['holidays_y']).all()
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_norec_only_static_feat(self):
        refining_params['rec_cold_start'] = False

        data_handler = DataHandler(base_data=base_data,
                                   static_features=static_features,
                                   global_dynamic_features=None,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        expected_target = pd.read_csv(REFINED_TARGET_PATH, sep=';')
        expected_static_data = pd.read_csv(REFINED_STATIC_PATH, sep=';')

        target, static_data, dynamic_data = data_handler.refining_specific()

        try:
            assert target.reset_index(drop=True).equals(expected_target.reset_index(drop=True))
            assert static_data.reset_index(drop=True).equals(expected_static_data.reset_index(drop=True))
            assert dynamic_data is None
        except AssertionError:
            pytest.fail("Test failed on nominal case.")


class DataHandlerDeepArFormatingTests:
    def test_nominal(self):
        # Expected
        with open(os.path.join(DATA_PATH, 'train_jsonline'), 'r') as f:
            expected_train_jsonline = f.read()
        df_expected_train_jsonline = pd.read_json(expected_train_jsonline,
                                                  orient='records',
                                                  lines=True
                                                  )
        df_expected_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_train_jsonline.reset_index(drop=True, inplace=True)
        with open(os.path.join(DATA_PATH, 'predict_jsonline'), 'r') as f:
            expected_predict_jsonline = f.read()
        df_expected_predict_jsonline = pd.read_json(expected_predict_jsonline,
                                                    orient='records',
                                                    lines=True
                                                    )
        df_expected_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_predict_jsonline.reset_index(drop=True, inplace=True)

        # Actual
        refining_params['rec_cold_start'] = True
        data_handler = DataHandler(base_data=base_data,
                                   static_features=static_features,
                                   global_dynamic_features=global_dynamic_features,
                                   specific_dynamic_features=specific_dynamic_features,
                                   **refining_params
                                   )

        data_handler.import_all_data()
        target = pd.read_csv(REFINED_TARGET_PATH, sep=';')
        static_data = pd.read_csv(REFINED_STATIC_PATH, sep=';')
        dynamic_data = pd.read_csv(REFINED_DYNAMIC_PATH, sep=';')

        data_handler.df_target, data_handler.df_static_data, data_handler.df_dynamic_data = \
            target, static_data, dynamic_data

        train_jsonline, predict_jsonline = data_handler.deepar_formatting(data_handler.df_target,
                                                                          data_handler.df_static_data,
                                                                          data_handler.df_dynamic_data
                                                                          )
        df_train_jsonline = pd.read_json(train_jsonline, orient='records', lines=True)
        df_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_train_jsonline.reset_index(drop=True, inplace=True)
        df_predict_jsonline = pd.read_json(predict_jsonline, orient='records', lines=True)
        df_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_predict_jsonline.reset_index(drop=True, inplace=True)

        try:
            assert df_expected_train_jsonline.equals(df_train_jsonline)
            assert df_expected_predict_jsonline.equals(df_predict_jsonline)
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_isrec_no_dynamic_feat(self):
        # Expected
        with open(os.path.join(DATA_PATH, 'train_jsonline_isrec_no_dynamic_feat'), 'r') as f:
            expected_train_jsonline = f.read()
        df_expected_train_jsonline = pd.read_json(expected_train_jsonline,
                                                  orient='records',
                                                  lines=True
                                                  )
        df_expected_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_train_jsonline.reset_index(drop=True, inplace=True)
        with open(os.path.join(DATA_PATH, 'predict_jsonline_isrec_no_dynamic_feat'), 'r') as f:
            expected_predict_jsonline = f.read()
        df_expected_predict_jsonline = pd.read_json(expected_predict_jsonline,
                                                    orient='records',
                                                    lines=True
                                                    )
        df_expected_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_predict_jsonline.reset_index(drop=True, inplace=True)

        # Actual
        refining_params['rec_cold_start'] = True
        data_handler = DataHandler(base_data=base_data,
                                   static_features=static_features,
                                   global_dynamic_features=None,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        data_handler.df_target, data_handler.df_static_data, data_handler.df_dynamic_data = \
            data_handler.refining_specific()

        train_jsonline, predict_jsonline = data_handler.deepar_formatting(data_handler.df_target,
                                                                          data_handler.df_static_data,
                                                                          data_handler.df_dynamic_data
                                                                          )
        df_train_jsonline = pd.read_json(train_jsonline, orient='records', lines=True)
        df_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_train_jsonline.reset_index(drop=True, inplace=True)
        df_predict_jsonline = pd.read_json(predict_jsonline, orient='records', lines=True)
        df_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_predict_jsonline.reset_index(drop=True, inplace=True)

        try:
            assert df_expected_train_jsonline.equals(df_train_jsonline)
            assert df_expected_predict_jsonline.equals(df_predict_jsonline)
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_isrec_no_static_dynamic_feat(self):
        # Expected
        with open(os.path.join(DATA_PATH, 'train_jsonline_isrec_no_static_dynamic_feat'), 'r') as f:
            expected_train_jsonline = f.read()
        df_expected_train_jsonline = pd.read_json(expected_train_jsonline,
                                                  orient='records',
                                                  lines=True
                                                  )
        df_expected_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_train_jsonline.reset_index(drop=True, inplace=True)
        with open(os.path.join(DATA_PATH, 'predict_jsonline_isrec_no_static_dynamic_feat'), 'r') as f:
            expected_predict_jsonline = f.read()
        df_expected_predict_jsonline = pd.read_json(expected_predict_jsonline,
                                                    orient='records',
                                                    lines=True
                                                    )
        df_expected_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_predict_jsonline.reset_index(drop=True, inplace=True)

        # Actual
        refining_params['rec_cold_start'] = True
        data_handler = DataHandler(base_data=base_data,
                                   static_features=None,
                                   global_dynamic_features=None,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        data_handler.df_target, data_handler.df_static_data, data_handler.df_dynamic_data = \
            data_handler.refining_specific()

        train_jsonline, predict_jsonline = data_handler.deepar_formatting(data_handler.df_target,
                                                                          data_handler.df_static_data,
                                                                          data_handler.df_dynamic_data
                                                                          )
        df_train_jsonline = pd.read_json(train_jsonline, orient='records', lines=True)
        df_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_train_jsonline.reset_index(drop=True, inplace=True)
        df_predict_jsonline = pd.read_json(predict_jsonline, orient='records', lines=True)
        df_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_predict_jsonline.reset_index(drop=True, inplace=True)

        try:
            assert df_expected_train_jsonline.equals(df_train_jsonline)
            assert df_expected_predict_jsonline.equals(df_predict_jsonline)
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_norec(self):
        # Expected
        with open(os.path.join(DATA_PATH, 'train_jsonline_norec'), 'r') as f:
            expected_train_jsonline = f.read()
        df_expected_train_jsonline = pd.read_json(expected_train_jsonline,
                                                  orient='records',
                                                  lines=True
                                                  )
        df_expected_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_train_jsonline.reset_index(drop=True, inplace=True)
        with open(os.path.join(DATA_PATH, 'predict_jsonline_norec'), 'r') as f:
            expected_predict_jsonline = f.read()
        df_expected_predict_jsonline = pd.read_json(expected_predict_jsonline,
                                                    orient='records',
                                                    lines=True
                                                    )
        df_expected_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_predict_jsonline.reset_index(drop=True, inplace=True)

        # Actual
        refining_params['rec_cold_start'] = False
        data_handler = DataHandler(base_data=base_data,
                                   static_features=static_features,
                                   global_dynamic_features=global_dynamic_features,
                                   specific_dynamic_features=specific_dynamic_features,
                                   **refining_params
                                   )

        data_handler.import_all_data()

        data_handler.df_target, data_handler.df_static_data, data_handler.df_dynamic_data = \
            data_handler.refining_specific()

        train_jsonline, predict_jsonline = data_handler.deepar_formatting(data_handler.df_target,
                                                                          data_handler.df_static_data,
                                                                          data_handler.df_dynamic_data
                                                                          )
        df_train_jsonline = pd.read_json(train_jsonline, orient='records', lines=True)
        df_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_train_jsonline.reset_index(drop=True, inplace=True)
        df_predict_jsonline = pd.read_json(predict_jsonline, orient='records', lines=True)
        df_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_predict_jsonline.reset_index(drop=True, inplace=True)

        try:
            assert df_expected_train_jsonline.equals(df_train_jsonline)
            assert df_expected_predict_jsonline.equals(df_predict_jsonline)
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_norec_no_dynamic_feat(self):
        # Expected
        with open(os.path.join(DATA_PATH, 'train_jsonline_isrec_no_dynamic_feat'), 'r') as f:
            expected_train_jsonline = f.read()
        df_expected_train_jsonline = pd.read_json(expected_train_jsonline,
                                                  orient='records',
                                                  lines=True
                                                  )
        df_expected_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_train_jsonline.reset_index(drop=True, inplace=True)
        with open(os.path.join(DATA_PATH, 'predict_jsonline_isrec_no_dynamic_feat'), 'r') as f:
            expected_predict_jsonline = f.read()
        df_expected_predict_jsonline = pd.read_json(expected_predict_jsonline,
                                                    orient='records',
                                                    lines=True
                                                    )
        df_expected_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_predict_jsonline.reset_index(drop=True, inplace=True)

        # Actual
        refining_params['rec_cold_start'] = False
        data_handler = DataHandler(base_data=base_data,
                                   static_features=static_features,
                                   global_dynamic_features=None,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        data_handler.df_target, data_handler.df_static_data, data_handler.df_dynamic_data = \
            data_handler.refining_specific()

        train_jsonline, predict_jsonline = data_handler.deepar_formatting(data_handler.df_target,
                                                                          data_handler.df_static_data,
                                                                          data_handler.df_dynamic_data
                                                                          )
        df_train_jsonline = pd.read_json(train_jsonline, orient='records', lines=True)
        df_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_train_jsonline.reset_index(drop=True, inplace=True)
        df_predict_jsonline = pd.read_json(predict_jsonline, orient='records', lines=True)
        df_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_predict_jsonline.reset_index(drop=True, inplace=True)

        try:
            assert df_expected_train_jsonline.equals(df_train_jsonline)
            assert df_expected_predict_jsonline.equals(df_predict_jsonline)
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

    def test_nominal_norec_no_static_dynamic_feat(self):
        # Expected
        with open(os.path.join(DATA_PATH, 'train_jsonline_isrec_no_static_dynamic_feat'), 'r') as f:
            expected_train_jsonline = f.read()
        df_expected_train_jsonline = pd.read_json(expected_train_jsonline,
                                                  orient='records',
                                                  lines=True
                                                  )
        df_expected_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_train_jsonline.reset_index(drop=True, inplace=True)
        with open(os.path.join(DATA_PATH, 'predict_jsonline_isrec_no_static_dynamic_feat'), 'r') as f:
            expected_predict_jsonline = f.read()
        df_expected_predict_jsonline = pd.read_json(expected_predict_jsonline,
                                                    orient='records',
                                                    lines=True
                                                    )
        df_expected_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_expected_predict_jsonline.reset_index(drop=True, inplace=True)

        # Actual
        refining_params['rec_cold_start'] = False
        data_handler = DataHandler(base_data=base_data,
                                   static_features=None,
                                   global_dynamic_features=None,
                                   specific_dynamic_features=None,
                                   **refining_params
                                   )

        data_handler.import_base_data()

        data_handler.df_target, data_handler.df_static_data, data_handler.df_dynamic_data = \
            data_handler.refining_specific()

        train_jsonline, predict_jsonline = data_handler.deepar_formatting(data_handler.df_target,
                                                                          data_handler.df_static_data,
                                                                          data_handler.df_dynamic_data
                                                                          )
        df_train_jsonline = pd.read_json(train_jsonline, orient='records', lines=True)
        df_train_jsonline.sort_values(by=['model_id'], inplace=True)
        df_train_jsonline.reset_index(drop=True, inplace=True)
        df_predict_jsonline = pd.read_json(predict_jsonline, orient='records', lines=True)
        df_predict_jsonline.sort_values(by=['model_id'], inplace=True)
        df_predict_jsonline.reset_index(drop=True, inplace=True)

        try:
            assert df_expected_train_jsonline.equals(df_train_jsonline)
            assert df_expected_predict_jsonline.equals(df_predict_jsonline)
        except AssertionError:
            pytest.fail("Test failed on nominal case.")
