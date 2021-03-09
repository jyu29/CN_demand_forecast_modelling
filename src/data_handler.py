import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import src.utils as ut
from src.data_cleaning import (check_weeks_df, generate_empty_dyn_feat_global,
                               cold_start_rec, pad_to_cutoff)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class data_handler:
    """
    Data Handler from refined data global to feature engineering for
    the demand Forecast project.

    Args:
    """

    def __init__(self,
                 cutoff: int,
                 params: dict,
                 static_data: dict,
                 global_dynamic_data: dict = None
                 # df_model_week_sales: pd.DataFrame = None,
                 # df_model_week_tree: pd.DataFrame = None,
                 # df_model_week_mrp: pd.DataFrame = None,
                 # df_dyn_feat_global: pd.DataFrame = None
                 ):

        self.cutoff = cutoff
        self.cat_cols = params['functional_parameters']['cat_cols']

        self.min_ts_len = params['functional_parameters']['min_ts_len']
        self.prediction_length = params['functional_parameters']['prediction_length']
        self.patch_covid = params['functional_parameters']['patch_covid']

        self.patch_covid_weeks = params['functional_parameters']['patch_covid_weeks']

        self.target_cluster_keys = params['functional_parameters']['target_cluster_keys']

        # Static data init
        for dataset in static_data.keys():
            assert isinstance(static_data[dataset], (str, pd.DataFrame)), "Value in dict `static_data` must be S3 URI or pd.DataFrame"
        self.static_data = static_data

        # Global dynamic data init
        if global_dynamic_data:
            for dataset in global_dynamic_data.keys():
                assert isinstance(global_dynamic_data[dataset], (str, pd.DataFrame)), "Value in dict `static_data` must be S3 URI or pd.DataFrame"
            self.global_dynamic_data = global_dynamic_data

        self.params = params
        self.refined_global_bucket = params['buckets']['refined_data_global']
        self.refined_specific_bucket = params['buckets']['refined_data_specific']
        self.path = {}
        self.path['train_path'] = params['functional_parameters']['train_path']
        self.path['predict_path'] = params['functional_parameters']['predict_path']

    def execute_data_refining_specific(self):
        """
        Data refining pipeline for specific data
        """

        # Static all data
        self.import_all_data()

        # Refining specific
        self.df_target, self.df_static_data, self.df_dynamic_data = self.refining_specific()

        # DeepAR Formatting
        self.train_jsonline, self.predict_jsonline = self.deepar_formatting(self.df_target, self.df_static_data, self.df_dynamic_data)

        # Saving jsonline files on S3
        ut.write_str_to_file_on_s3(self.train_jsonline, self.bucket['refined_data_specific'], self.path['train_path'])
        ut.write_str_to_file_on_s3(self.predict_jsonline, self.bucket['refined_data_specific'], self.path['train_path'])

    def import_all_data(self):
        # Static data import
        logger.debug("Starting static data import")
        self.import_static_data()
        logger.debug("Static data imported done.")

        # Global dynamic data import
        logger.debug("Starting global dynamic data import")
        self.import_global_dynamic_data()
        logger.debug("Global dynamic data imported done.")

        # Specific dynamic data import
        logger.debug("Starting specific dynamic data import")
        self.import_specific_dynamic_data()
        logger.debug("Global specific data imported done.")

        # Specific dynamic data import
        logger.debug("Starting specific dynamic data import")
        self.import_specific_dynamic_data()
        logger.debug("Global specific data imported done.")

    def refining_specific(self):
        # Sales refining
        df_sales = self.static_data['model_week_sales']
        df_sales.loc[:, 'date'] = pd.to_datetime(df_sales.loc[:, 'date'])
        df_sales = df_sales[df_sales['week_id'] < self.cutoff]
        df_sales.rename(columns={'sales_quantity': 'y'}, inplace=True)
        self.static_data['model_week_sales'] = df_sales

        # MRP refining
        df_mrp = self.static_data['model_week_mrp']
        df_mrp = df_mrp[df_mrp['week_id'] == self.cutoff]
        self.static_data['model_week_mrp'] = df_mrp

        # Tree refining
        df_tree = self.static_data['model_week_tree']
        df_tree = df_tree[df_tree['week_id'] == self.cutoff]
        self.static_data['model_week_tree'] = df_tree

        # Limiting Sales data to MRP active models
        df_sales = pd.merge(df_sales, df_mrp.loc[df_mrp['is_mrp_active'], ['model_id']])

        # Pad to cutoff Sales data
        df_sales = pad_to_cutoff(df_sales, self.cutoff)

        # Cold start reconstruction
        df_sales = cold_start_rec(df_sales,
                                  self.static_data['model_week_sales'],
                                  df_tree,
                                  self.min_ts_len,
                                  self.patch_covid_weeks,
                                  self.target_cluster_keys,
                                  self.patch_covid)

        # Creating df_target
        df_target = df_sales[['model_id', 'week_id', 'y']]

        # Creating df_static_data
        df_static_data = df_tree[['model_id'] + self.cat_cols]

        # Creating df_dynamic_data
        min_week = df_sales['week_id'].min()
        df_dynamic_data = generate_empty_dyn_feat_global(df_target,
                                                         min_week=min_week,
                                                         cutoff=self.cutoff,
                                                         future_projection=self.prediction_length
                                                         )

        # Adding is_rec dynamic feat
        df_is_rec = df_sales[['model_id', 'date', 'week_id', 'is_rec']]
        models = df_is_rec['model_id'].unique()
        dates = pd.date_range(start=ut.week_id_to_date(self.cutoff), periods=self.prediction_length, freq='W')
        m, d = pd.core.reshape.util.cartesian_product([models, dates])
        df_is_rec_future = pd.DataFrame({"model_id": m, "date": d})
        df_is_rec_future['week_id'] = ut.date_to_week_id(df_is_rec_future['date'])
        df_is_rec_future['is_rec'] = 0
        df_is_rec = df_is_rec.append(df_is_rec_future)
        df_is_rec = df_is_rec[['model_id', 'week_id', 'is_rec']]

        df_dynamic_data = self._add_dyn_feat(df_dynamic_data,
                                             df_feat=df_is_rec,
                                             min_week=min_week,
                                             cutoff=self.cutoff,
                                             future_weeks=self.prediction_length)

        # Adding provided dynamic features
        if self.global_dynamic_data:
            for dataset in self.global_dynamic_data.keys():
                df_dynamic_data = self._add_dyn_feat(df_dynamic_data,
                                                     df_feat=self.global_dynamic_data[dataset],
                                                     min_week=min_week,
                                                     cutoff=self.cutoff,
                                                     future_weeks=self.prediction_length)

        return df_target, df_static_data, df_dynamic_data

    def deepar_formatting(self, df_target, df_static_data, df_dynamic_data):
        # Label Encode Categorical features
        for c in self.cat_cols:
            le = LabelEncoder()
            df_static_data[c] = le.fit_transform(df_static_data[c])
        df_static_data['cat'] = df_static_data[self.cat_cols].values.tolist()

        # Building df_predict
        # Adding prediction weeks necessary for dynamic features in df_predict
        df_predict = self._add_future_weeks(df_target).merge(df_dynamic_data, on=['model_id', 'week_id'], how='left')
        df_predict.sort_values(by=['model_id', 'week_id'], ascending=True, inplace=True)
        # Building data `start` & `target`
        df_predict = df_predict.groupby(by=['model_id'], sort=False).agg(start=('week_id', lambda x: ut.week_id_to_date(x.min()).strftime('%Y-%m-%d %H:%M:%S')),
                                                                         target=('y', lambda x: list(x.dropna())))
        # Adding categorical features
        df_predict = df_predict.merge(df_static_data[['model_id', 'cat']], left_index=True, right_on='model_id').set_index('model_id')
        # Concatenating dynamic features in list format
        df_dynamic_data_predict = df_dynamic_data.sort_values(by=['model_id', 'week_id'], ascending=True)\
            .groupby(by=['model_id'], sort=False)\
            .agg(is_rec=('is_rec', list),
                 perc_store_open=('perc_store_open', list))
        df_dynamic_data_predict['dynamic_feat'] = df_dynamic_data_predict.values.tolist()
        # Adding dynamic features
        df_predict = df_predict.merge(df_dynamic_data_predict[['dynamic_feat']], left_index=True, right_index=True, how='left')
        df_predict.reset_index(inplace=True)

        # Building df_train
        # Limiting dataset to avoid any future data
        df_train = df_target[df_target['week_id'] < self.cutoff]
        df_dynamic_data_train = df_dynamic_data[df_dynamic_data['week_id'] < self.cutoff]
        # Building data `start` & `target`
        df_train.sort_values(by=['model_id', 'week_id'], ascending=True, inplace=True)
        df_train = df_train.groupby(by=['model_id'], sort=False).agg(start=('week_id', lambda x: ut.week_id_to_date(x.min()).strftime('%Y-%m-%d %H:%M:%S')),
                                                                     target=('y', lambda x: list(x.dropna())))
        # Adding categorical features
        df_train = df_train.merge(df_static_data[['model_id', 'cat']], left_index=True, right_on='model_id').set_index('model_id')
        # Concatenating dynamic features in list format
        df_dynamic_data_train = df_dynamic_data_train.sort_values(by=['model_id', 'week_id'], ascending=True)\
            .groupby(by=['model_id'], sort=False)\
            .agg(is_rec=('is_rec', list),
                 perc_store_open=('perc_store_open', list))
        df_dynamic_data_train['dynamic_feat'] = df_dynamic_data_train.values.tolist()
        # Adding dynamic features
        df_train = df_train.merge(df_dynamic_data_train[['dynamic_feat']], left_index=True, right_index=True, how='left')
        # Shuffling df_train
        df_train = df_train.sample(frac=1)
        df_train.reset_index(inplace=True)

        # Converting to jsonline
        train_jsonline = df_train.to_json(orient='records', lines=True)
        predict_jsonline = df_predict.to_json(orient='records', lines=True)

        # Checking jsonline datasets
        self.check_json_line(train_jsonline)
        self.check_json_line(predict_jsonline, future_proj_len=52)

        return train_jsonline, predict_jsonline

    def import_static_data(self):
        for dataset in self.static_data.keys():
            if isinstance(self.static_data[dataset], str):
                logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                s3_uri = self.static_data[dataset]
                bucket, path = ut.from_uri(s3_uri)
                self.static_data[dataset] = ut.read_multipart_parquet_s3(bucket, path)
                logger.debug(f"Dataset {dataset} imported from S3.")

    def import_global_dynamic_data(self):
        for dataset in self.global_dynamic_data.keys():
            if isinstance(self.global_dynamic_data[dataset], str):
                logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                bucket, path = ut.from_uri(self.static_data[dataset])
                self.global_dynamic_data[dataset] = ut.read_multipart_parquet_s3(bucket, path)
                logger.debug(f"Dataset {dataset} imported from S3.")

    def import_specific_dynamic_data(self):
        pass

    def _add_dyn_feat(self, df_dynamic_data, df_feat, min_week, cutoff, future_weeks, week_column='week_id'):
        # Checks
        check_weeks_df(df_feat, min_week, cutoff, future_weeks, week_column=week_column)

        if 'model_id' not in df_feat.columns:
            # assert models is not None, "If `df_feat` is a global dynamic feature, you must provide `models` with a pd.DataFrame of expected models with 'model_id' column"
            models = pd.DataFrame({'model_id': df_dynamic_data['model_id'].unique()})
            df_feat = df_feat.merge(models, how='cross')

        # Checking if `model_id` and `week_id` lists match in both df_dynamic_data and df_feat
        assert len(set(df_dynamic_data['model_id'].unique()) - set(df_feat['model_id'].unique())) == 0,\
            "Mismatch in model_id list between the dynamic feature and the sales dataset"
        assert len(set(df_dynamic_data['week_id'].unique()) - set(df_feat['week_id'].unique())) == 0,\
            "Mismatch in week_id list between the dynamic feature and the sales dataset"

        df_with_new_feat = pd.merge(df_dynamic_data, df_feat, on=['week_id', 'model_id'])

        return df_with_new_feat

    def _add_future_weeks(self, df):
        cutoff = self.cutoff
        prediction_length = self.prediction_length
        model_ids = df['model_id'].unique()

        future_date_range = pd.date_range(start=ut.week_id_to_date(cutoff), periods=prediction_length, freq='W')
        future_date_range_weeks = [ut.date_to_week_id(w) for w in future_date_range]

        w, m = pd.core.reshape.util.cartesian_product([future_date_range_weeks, model_ids])

        future_model_week = pd.DataFrame({'model_id': m, 'week_id': w})

        df = df.append(future_model_week)

        return df

    def check_json_line(self, jsonline, future_proj_len=0):
        df = pd.read_json(jsonline, orient='records', lines=True)

        # Test if target >= min_ts_len
        df['target_len'] = df.apply(lambda x: len(x['target']), axis=1)
        test = df['target_len'] >= self.min_ts_len
        assert all(test.values), 'Some models have a `target` less than `min_ts_len`'

        # Test if target length is right
        df['target_len_test'] = df.apply(lambda x: ut.date_to_week_id(pd.to_datetime(x['start']) + pd.Timedelta(x['target_len'], 'W')) == self.cutoff, axis=1)
        assert all(df['target_len_test'].values), "Some models have a 'target' length which doesn't match with the 'start' date"

        if 'cat' in df.columns:
            # Test if right number of categorical features
            df['cat_feat_nb'] = df.apply(lambda x: len(x['cat']), axis=1)
            test = df['cat_feat_nb'] == len(self.cat_cols)
            assert all(test.values), "Some models don't have the right number of categorical features"

        if 'dynamic_feat' in df.columns:
            nb_dyn_feat = len(self.global_dynamic_data)
            # Adding one dynamic feature as `is_rec` is defined during refining
            nb_dyn_feat += 1

            # Test if right number of dynamic features
            df['dyn_feat_nb'] = df.apply(lambda x: len(x['dynamic_feat']), axis=1)
            test = df[['dyn_feat_nb']] == nb_dyn_feat
            assert all(test.values), "Some models don't have the right number of dynamic features"

            # Test if right length of dynamic features
            test_dict = {}
            for i in range(nb_dyn_feat):
                df[f'dyn_feat_{i}_len'] = df.apply(lambda x: len(x['dynamic_feat'][i]), axis=1)
                df[f'dyn_feat_{i}_len_test'] = df[f'dyn_feat_{i}_len'] == df['target_len'] + future_proj_len
                test_dict[i] = f'dyn_feat_{i}_len_test'

            for i in test_dict.keys():
                assert df[df[test_dict[i]]].shape[0] == df.shape[0], \
                    f"Some models don't have the right dynamic feature length for feature {i}"
