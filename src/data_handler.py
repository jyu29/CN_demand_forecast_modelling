import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import src.utils as ut
from src.data_cleaning import (check_weeks_df, generate_empty_dyn_feat_global,
                               history_reconstruction, pad_to_cutoff)

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

        # # Data Import
        # if self.df_model_week_sales is None:
        #     self.df_model_week_sales = self._generate_model_week_sales()

        # if self.df_model_week_tree is None:
        #     self.df_model_week_tree = self._generate_model_week_tree()

        # if self.df_model_week_mrp is None:
        #     self.df_model_week_mrp = self._generate_model_week_mrp()

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

        # Train/Predict split
        self.df_train, self.df_predict = self._generate_target_data()

    def import_static_data(self):
        for dataset in self.static_data.keys():
            if all([dataset == 'model_week_sales', isinstance(self.static_data[dataset], str)]):
                logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                self.static_data[dataset] = self._generate_model_week_sales(self.static_data[dataset])
                logger.debug(f"Dataset {dataset} imported from S3.")
            elif all([dataset == 'model_week_tree', isinstance(self.static_data[dataset], str)]):
                logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                self.static_data[dataset] = self._generate_model_week_tree(self.static_data[dataset])
                logger.debug(f"Dataset {dataset} imported from S3.")
            elif all([dataset == 'model_week_mrp', isinstance(self.static_data[dataset], str)]):
                logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                self.static_data[dataset] = self._generate_model_week_mrp(self.static_data[dataset])
                logger.debug(f"Dataset {dataset} imported from S3.")

    def import_global_dynamic_data(self):
        if self.global_dynamic_data:
            # Dynamic Global features generation/import
            min_week = self.static_data['model_week_sales']['week_id'].min()
            self.df_dyn_feat_global = generate_empty_dyn_feat_global(min_week=min_week, cutoff=self.cutoff, future_projection=self.prediction_length)

            # Adding dynamic global features one by one
            for dataset in self.global_dynamic_data.keys():
                if isinstance(self.static_data[dataset], str):
                    logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                    bucket, path = ut.from_uri(self.static_data[dataset])
                    df = ut.read_multipart_parquet_s3(bucket, path)
                    self.df_dyn_feat_global = self._add_dyn_feat_global(self.df_dyn_feat_global,
                                                                        df_feat=df,
                                                                        min_week=min_week,
                                                                        cutoff=self.cutoff,
                                                                        future_weeks=self.prediction_length)
                    logger.debug(f"Dataset {dataset} imported from S3.")

    def import_specific_dynamic_data(self):
        pass

    def _generate_model_week_sales(self, s3_uri):
        bucket, path = ut.from_uri(s3_uri)
        df_model_week_sales = ut.read_multipart_parquet_s3(bucket, path)
        df_model_week_sales = df_model_week_sales[df_model_week_sales['week_id'] < self.cutoff]
        df_model_week_sales.rename(columns={'date': 'ds', 'sales_quantity': 'y'}, inplace=True)

        return df_model_week_sales

    def _generate_model_week_tree(self, s3_uri):
        bucket, path = ut.from_uri(s3_uri)
        df_model_week_tree = ut.read_multipart_parquet_s3(bucket, path)
        df_model_week_tree = ut.read_multipart_parquet_s3(self.refined_global_bucket, self.paths['model_week_tree'])
        df_model_week_tree = df_model_week_tree[df_model_week_tree['week_id'] == self.cutoff]

        return df_model_week_tree

    def _generate_model_week_mrp(self, s3_uri):
        bucket, path = ut.from_uri(s3_uri)
        df_model_week_mrp = ut.read_multipart_parquet_s3(bucket, path)
        df_model_week_mrp = ut.read_multipart_parquet_s3(self.refined_global_bucket, self.paths['model_week_mrp'])
        df_model_week_mrp = df_model_week_mrp[df_model_week_mrp['week_id'] == self.cutoff]

        return df_model_week_mrp

    def _generate_target_data(self):
        # List MRP valid models
        df_mrp_valid_model = self.df_model_week_mrp.loc[self.df_model_week_mrp['is_mrp_active'], ['model_id']]

        # Create df_train
        df_train = pd.merge(self.df_model_week_sales, df_mrp_valid_model)  # mrp valid filter
        df_train = pad_to_cutoff(df_train, self.cutoff)          # pad sales to cutoff

        # Rec histo
        df_train = history_reconstruction(df_train,
                                          self.df_model_week_sales,
                                          self.df_model_week_tree,
                                          self.min_ts_len,
                                          self.patch_covid_weeks,
                                          self.target_cluster_keys,
                                          self.patch_covid)

        # Add and encode cat features
        df_train = pd.merge(df_train, self.df_model_week_tree[['model_id'] + self.cat_cols])

        for c in self.cat_cols:
            le = LabelEncoder()
            df_train[c] = le.fit_transform(df_train[c])

        return df_train, df_train

    def _add_dyn_feat_global(self, df_dyn_feat_global, df_feat, min_week, cutoff, future_weeks, week_column='week_id'):
        check_weeks_df(df_feat, min_week, cutoff, future_weeks, week_column=week_column)

        df_with_new_feat = pd.merge(df_dyn_feat_global, df_feat, left_index=True, right_on=week_column)

        return df_with_new_feat

    def _generate_store_openings(self, path=None):
        if path is None:
            path = self.paths['store_openings']
        df_store_openings = ut.read_multipart_parquet_s3(self.refined_global_bucket, path)

        return df_store_openings
