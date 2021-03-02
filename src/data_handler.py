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
        self.prediction_length = params['functional_parameters']['hyperparameters']['prediction_length']
        self.min_ts_len = params['functional_parameters']['min_ts_len']
        self.patch_covid_weeks = params['functional_parameters']['patch_covid_weeks']
        self.target_cluster_keys = params['functional_parameters']['target_cluster_keys']
        self.patch_covid = params['functional_parameters']['patch_covid']

        # self.df_model_week_sales = df_model_week_sales
        # self.df_model_week_tree = df_model_week_tree
        # self.df_model_week_mrp = df_model_week_mrp

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
        # self.paths = {'model_week_sales': f"{params['paths']['refined_global_path']}model_week_sales",
        #               'model_week_tree': f"{params['paths']['refined_global_path']}model_week_tree",
        #               'model_week_mrp': f"{params['paths']['refined_global_path']}model_week_mrp",
        #               'store_openings': f"{params['paths']['refined_global_path']}store_openings"
        #               }

    def execute_data_refining_specific(self):
        """
        Data refining pipeline for specific data
        """

        # Static all data
        self.import_all_data()

        # Train/Predict split
        self.df_train, self.df_predict = self._generate_target_data()

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

    def import_static_data(self):
        for dataset in self.static_data.keys():
            # Model Week Sales import
            if dataset == 'model_week_sales':
                if isinstance(self.static_data[dataset], str):
                    logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                    s3_uri = self.static_data[dataset]
                    bucket, path = ut.from_uri(s3_uri)
                    df = ut.read_multipart_parquet_s3(bucket, path)
                    logger.debug(f"Dataset {dataset} imported from S3.")
                else:
                    df = self.static_data[dataset].copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['week_id'] < self.cutoff]
                df.rename(columns={'sales_quantity': 'y'}, inplace=True)
                self.static_data[dataset] = df
            # Model Week Tree import
            elif any([dataset == 'model_week_tree', dataset == 'model_week_mrp']):
                if isinstance(self.static_data[dataset], str):
                    logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                    s3_uri = self.static_data[dataset]
                    bucket, path = ut.from_uri(s3_uri)
                    df = ut.read_multipart_parquet_s3(bucket, path)
                    logger.debug(f"Dataset {dataset} imported from S3.")
                else:
                    df = self.static_data[dataset].copy()
                df = df[df['week_id'] == self.cutoff]
                self.static_data[dataset] = df

    def import_global_dynamic_data(self):
        if self.global_dynamic_data:
            # Dynamic Global features generation/import
            min_week = self.static_data['model_week_sales']['week_id'].min()
            self.df_dyn_feat_global = generate_empty_dyn_feat_global(min_week=min_week, cutoff=self.cutoff, future_projection=self.prediction_length)

            # Adding dynamic global features one by one
            for dataset in self.global_dynamic_data.keys():
                if isinstance(self.global_dynamic_data[dataset], str):
                    logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                    bucket, path = ut.from_uri(self.static_data[dataset])
                    df = ut.read_multipart_parquet_s3(bucket, path)
                    logger.debug(f"Dataset {dataset} imported from S3.")
                else:
                    df = self.global_dynamic_data[dataset]
                self.df_dyn_feat_global = self._add_dyn_feat_global(self.df_dyn_feat_global,
                                                                    df_feat=df,
                                                                    min_week=min_week,
                                                                    cutoff=self.cutoff,
                                                                    future_weeks=self.prediction_length)

    def import_specific_dynamic_data(self):
        pass

    def _generate_target_data(self):
        # List MRP valid models
        df_mrp_valid_model = self.static_data['model_week_mrp']
        df_mrp_valid_model = df_mrp_valid_model.loc[df_mrp_valid_model['is_mrp_active'], ['model_id']]

        # Create df_predict
        df_predict = pd.merge(self.static_data['model_week_sales'], df_mrp_valid_model)  # mrp valid filter
        df_predict = pad_to_cutoff(df_predict, self.cutoff)          # pad sales to cutoff

        # Rec histo
        df_predict = cold_start_rec(df_predict,
                                    self.static_data['model_week_sales'],
                                    self.static_data['model_week_tree'],
                                    self.min_ts_len,
                                    self.patch_covid_weeks,
                                    self.target_cluster_keys,
                                    self.patch_covid)

        # Add future weeks to df_predict
        df_predict = self._add_future_weeks(df_predict)

        # Add and encode cat features
        df_predict = pd.merge(df_predict, self.static_data['model_week_tree'][['model_id'] + self.cat_cols])

        for c in self.cat_cols:
            le = LabelEncoder()
            df_predict[c] = le.fit_transform(df_predict[c])

        # Create df_train from df_predict
        df_train = df_predict[df_predict['week_id'] < self.cutoff]

        return df_train, df_predict

    def _add_dyn_feat_global(self, df_dyn_feat_global, df_feat, min_week, cutoff, future_weeks, week_column='week_id'):
        check_weeks_df(df_feat, min_week, cutoff, future_weeks, week_column=week_column)

        df_with_new_feat = pd.merge(df_dyn_feat_global, df_feat, left_index=True, right_on=week_column)

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
