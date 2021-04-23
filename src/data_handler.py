import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import src.utils as ut
from src.refining_specific_functions import (check_weeks_df, generate_empty_dyn_feat_global,
                                             cold_start_rec, pad_to_cutoff, is_rec_feature_processing,
                                             features_forward_fill, apply_first_lockdown_patch, zero_padding_rec)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def import_refining_config(environment: str,
                           cutoff: int,
                           run_name: str,
                           train_path: str,
                           predict_path: str
                           ) -> dict:
    """Handler to import specific refining configuration from YML file

    Args:
        environment (str): Set of parameters on which to load the parameters
        cutoff (int): Cutoff week in format YYYYWW (ISO 8601)
        run_name (str): Custom name for the current name - will propagate to saved files names
        df_jobs (pd.DataFrame): helper to ensure Sagemaker tracking of training & inference, and associated
            files paths.

    Returns:
        A dictionary with all parameters for specific refining process
    """
    assert isinstance(environment, str)
    assert isinstance(cutoff, int)
    assert isinstance(run_name, str)
    assert isinstance(train_path, str)
    assert isinstance(predict_path, str)

    params_full_path = f"config/{environment}.yml"
    params = ut.read_yml(params_full_path)

    refining_params = {'cutoff': cutoff,
                       'patch_first_lockdown': params['refining_specific_parameters']['patch_first_lockdown'],
                       'rec_cold_start': params['refining_specific_parameters']['rec_cold_start'],
                       'rec_length': params['refining_specific_parameters']['rec_length'],
                       'rec_cold_start_group': params['refining_specific_parameters']['rec_cold_start_group'],
                       'prediction_length': params['modeling_parameters']['hyperparameters']['prediction_length'],
                       'refined_global_bucket': params['buckets']['refined_data_global'],
                       'refined_specific_bucket': params['buckets']['refined_data_specific'],
                       'output_paths': {'train_path': train_path,
                                        'predict_path': predict_path
                                        }
                       }
    return refining_params


class DataHandler:
    """
    Data Handler from refined data global to feature engineering for
    the demand Forecast project.
    """

    def __init__(self,
                 cutoff: int,
                 base_data: dict,
                 patch_first_lockdown: bool,
                 rec_cold_start: bool,
                 rec_length: int,
                 prediction_length: int,
                 rec_cold_start_group: list,
                 refined_global_bucket: str,
                 refined_specific_bucket: str,
                 output_paths: dict,
                 static_features: dict = None,
                 global_dynamic_features: dict = None,
                 specific_dynamic_features: dict = None
                 ):
        """Instanciation of data handler.

        The Data Handler takes care of Data Refining for specific aspects of DeepAR algorithm,
        formats the data to the expected format, and save the results in AWS S3.

        Args:
            cutoff (int): Cutoff week in format YYYYWW (ISO 8601)
            static_data (dict): Dictionnary of pd.DataFrame or S3 URIs for static data
            rec_length (int): Minimum weeks expected in each input time series
            prediction_length (int): Number of forecasted weeks in the future
            cat_cols (list): List of `str` to select static columns expected in model_week_tree
            rec_cold_start_group (list): for the cold start reconstruction, columns to use in model_week_tree
                                    for the group average
            refined_global_bucket (str): S3 bucket on which the refined global data should be downloaded
            refined_specific_bucket (str): S3 bucket on which the refined specific data should be uploaded
            output_paths (dict): Dictionnary of S3 URIs for dynamic data (without bucket, including file name
                                    & extension) for the train  & predict JSON output files
            global_dynamic_data (dict)(optional): Dictionnary of pd.DataFrame or S3 URIs for dynamic data
        """
        self.cutoff = cutoff

        self.patch_first_lockdown = patch_first_lockdown
        self.rec_cold_start = rec_cold_start
        self.rec_length = rec_length
        self.rec_cold_start_group = rec_cold_start_group

        self.prediction_length = prediction_length

        if patch_first_lockdown:
            assert 'imputed_sales_lockdown_1' in base_data.keys(),\
                "Patching first lockdown requested, but imputation dataset not provided in base_data"

        # Base data init
        for dataset in base_data.keys():
            assert isinstance(base_data[dataset], (str, pd.DataFrame)), \
                "Value in dict `base_data` must be S3 URI or pd.DataFrame"
        self.base_data = base_data

        # Static features init
        self.static_features = None
        if static_features:
            for dataset in static_features.keys():
                assert isinstance(static_features[dataset], (str, pd.DataFrame)), \
                    "Value in dict `static_features` must be S3 URI or pd.DataFrame"
            self.static_features = static_features

        # Global dynamic data init
        self.global_dynamic_features = None
        if global_dynamic_features:
            for dataset in global_dynamic_features.keys():
                assert isinstance(global_dynamic_features[dataset]['dataset'], (str, pd.DataFrame)), \
                    "Value in dict `global_dynamic_features` must be S3 URI or pd.DataFrame"
            self.global_dynamic_features = {key: global_dynamic_features[key]['dataset']
                                            for key in global_dynamic_features}
            self.global_dynamic_features_projection = {key: global_dynamic_features[key]['projection']
                                                       for key in global_dynamic_features}

        # Specific dynamic data init
        self.specific_dynamic_features = None
        if specific_dynamic_features:
            for dataset in specific_dynamic_features.keys():
                assert isinstance(specific_dynamic_features[dataset]['dataset'], (str, pd.DataFrame)), \
                    "Value in dict `specific_dynamic_features` must be S3 URI or pd.DataFrame"
            self.specific_dynamic_features = {key: specific_dynamic_features[key]['dataset']
                                              for key in specific_dynamic_features}
            self.specific_dynamic_features_projection = {key: specific_dynamic_features[key]['projection']
                                                         for key in specific_dynamic_features}

        # Output json line datasets init
        assert 'train_path' in output_paths, "Output paths must include `train_path`"
        assert 'predict_path' in output_paths, "Output paths must include `predict_path`"
        for output in output_paths:
            assert isinstance(output_paths[output], (str)), "Output paths for jsonline files must be `str`"
        self.output_paths = output_paths

        logger.info(f"Data refining specific for Demand Forecast initialized for cutoff {self.cutoff}")
        if self.patch_first_lockdown:
            logger.info("Patch first lockdown of 2020 requested")
        if self.rec_cold_start:
            logger.info(f"Cold Start Reconstruction requested with {self.rec_length} minimum "
                        f"weeks and average on values {self.rec_cold_start_group}")
        else:
            logger.info("Cold Start Reconstruction not requested, a simple zero padding will be applied")

        logger.info(f"Expected prediction length is {self.prediction_length}")

    def execute_data_refining_specific(self):
        """Workflow method for data refining pipeline for specific data

        Executes all steps for data refining specific to Demand Forecast :
        * Data import (including optional S3 download & potential forward fill)
        * Data refining specific to Demand Forecast, including :
          * Dataset limitation to instance cutoff
          * Pad to cutoff for missing data (models for which sales stopped before the instance cutoff)
          * Optional cold start reconstruction (depending on parameters) and addition to dynamic features
          * static & dynamic features concatenation
        * Formatting datasets to the algorithm expected scheme
        * Upload to S3
        """

        # Static all data
        self.import_all_data()

        # Refining specific
        logger.info("Starting Data Refining...")
        self.df_target, self.df_static_data, self.df_dynamic_data = self.refining_specific()

        # DeepAR Formatting
        logger.info("Starting DeepAR formatting...")
        self.train_jsonline, self.predict_jsonline = self.deepar_formatting(self.df_target,
                                                                            self.df_static_data,
                                                                            self.df_dynamic_data
                                                                            )

        # Saving jsonline files on S3
        train_bucket, train_path = ut.from_uri(self.output_paths['train_path'])
        predict_bucket, predict_path = ut.from_uri(self.output_paths['predict_path'])
        ut.write_str_to_file_on_s3(self.train_jsonline, train_bucket, train_path)
        logger.info(f"Train jsonline file saved at {self.output_paths['train_path']}")
        ut.write_str_to_file_on_s3(self.predict_jsonline, predict_bucket, predict_path)
        logger.info(f"Predict jsonline file saved at {self.output_paths['predict_path']}")

    def import_all_data(self):
        """Workflow method to integrate datasets.

        Provides a workflow to integrate base data (used for cold start reconstruction for example), static features,
        Global dynamic features (temporal features not specific to a model), and specific dynamic features (temporal
        features defined at model level)
        """
        # Base data import
        self.import_base_data()
        logger.info("Attribute `base_data` created.")

        # Static features import
        if self.static_features:
            self.import_static_features()
            logger.info("Attribute `static_features` created")
        else:
            logger.info("No static features specified.")

        # Global dynamic features import
        if self.global_dynamic_features:
            self.import_global_dynamic_features()
            logger.info("Attribute `global_dynamic_features` created")
        else:
            logger.info("No global dynamic features specified.")

        # Specific dynamic features import
        if self.specific_dynamic_features:
            self.import_specific_dynamic_features()
            logger.info("Attribute `specific_dynamic_features` created")
        else:
            logger.info("No specific dynamic features specified.")

    def refining_specific(self):
        """Workflow method for data refining specific to Demand Forecast

        Processes base data, static features, global dynamic features & specific dynamic features to return
        concatenated and refined datasets (target, static features & dynamic features), with several format checks.
        Steps include:
          * Dataset limitation to instance cutoff
          * Pad to cutoff for missing data (models for which sales stopped before the instance cutoff)
          * Optional cold start reconstruction (depending on parameters) and addition to dynamic features
          * static & dynamic features concatenation

        Returns:
            df_target (pd.DataFrame): Temporal & model-specific dataset including target values up to instance cutoff
            df_static_features (pd.DataFrame): Model-specific dataset with static labels
            df_dynamic_features (pd.DataFrame): Temporal & model-specific dataset with dynamic values
        """
        # Init
        df_static_features = None
        df_dynamic_features = None

        # Sales refining
        df_sales = self.base_data['model_week_sales']
        df_sales = df_sales[df_sales['week_id'] < self.cutoff]
        if self.patch_first_lockdown:
            df_sales = apply_first_lockdown_patch(df_sales, self.base_data['imputed_sales_lockdown_1'])
        self.base_data['model_week_sales'] = df_sales
        logger.debug(f"Limited sales data up to cutoff {self.cutoff}")

        # MRP refining
        df_mrp = self.base_data['model_week_mrp']
        df_mrp = df_mrp[df_mrp['week_id'] == self.cutoff]
        self.base_data['model_week_mrp'] = df_mrp
        logger.debug(f"Limited MRP data to cutoff {self.cutoff}")

        # Tree refining
        df_tree = self.base_data['model_week_tree']
        df_tree = df_tree[df_tree['week_id'] == self.cutoff]
        self.base_data['model_week_tree'] = df_tree
        logger.debug(f"Limited tree data to cutoff {self.cutoff}")

        # Limiting Sales data to MRP active models
        df_sales = pd.merge(df_sales, df_mrp.loc[df_mrp['is_mrp_active'], ['model_id']])
        logger.debug("Sales data filtered on active MRP status")

        # Pad to cutoff Sales data
        logger.debug("Starting pad to cutoff...")
        df_sales = pad_to_cutoff(df_sales, self.cutoff)
        logger.debug("Pad to cutoff done")

        # Cold start reconstruction
        if self.rec_cold_start:
            logger.debug("Cold start reconstruction requested. Starting reconstruction...")
            df_sales = cold_start_rec(df_sales,
                                      self.base_data['model_week_sales'],
                                      self.base_data['model_week_tree'],
                                      self.rec_length,
                                      self.rec_cold_start_group)
            logger.debug("Cold start reconstruction done.")
        # Zero padding reconstruction
        else:
            logger.debug("Zero padding reconstruction requested. Starting reconstruction...")
            df_sales = zero_padding_rec(df_sales, self.base_data['model_week_sales'], self.rec_length)
            logger.debug("Zero padding reconstruction done.")

        # Creating df_target
        df_target = df_sales[['model_id', 'week_id', 'sales_quantity']]

        # Creating df_static_features
        if self.static_features:
            df_static_features = df_target[['model_id']].drop_duplicates().copy()
            for dataset in self.static_features.keys():
                df_static_features = self._add_static_feat(df_static_features,
                                                           df_feat=self.static_features[dataset])
                logger.debug(f"Added static feature {dataset} to `df_static_features`")

        # Creating df_dynamic_features
        min_week = df_sales['week_id'].min()
        if any([self.rec_cold_start,
                self.global_dynamic_features,
                self.specific_dynamic_features
                ]
               ):
            df_dynamic_features = generate_empty_dyn_feat_global(df_target,
                                                                 min_week=min_week,
                                                                 cutoff=self.cutoff,
                                                                 future_projection=self.prediction_length
                                                                 )

        # Building is_rec specific dynamic feature and adding it to the dynamic features
        if self.rec_cold_start:
            df_is_rec = is_rec_feature_processing(df_sales, self.cutoff, self.prediction_length)
            df_dynamic_features = self._add_dyn_feat(df_dynamic_features,
                                                     df_feat=df_is_rec,
                                                     min_week=min_week,
                                                     cutoff=self.cutoff,
                                                     future_weeks=self.prediction_length)
            logger.debug("Cold start reconstruction requested. Added reconstruction to `df_dynamic_features`.")

        # Adding provided global dynamic features
        if self.global_dynamic_features:
            for dataset in self.global_dynamic_features.keys():
                df_dynamic_features = self._add_dyn_feat(df_dynamic_features,
                                                         df_feat=self.global_dynamic_features[dataset],
                                                         min_week=min_week,
                                                         cutoff=self.cutoff,
                                                         future_weeks=self.prediction_length)
                logger.debug(f"Added global dynamic feature {dataset} to `df_dynamic_features`")

        # Adding provided specific dynamic features
        if self.specific_dynamic_features:
            for dataset in self.specific_dynamic_features.keys():
                df_dynamic_features = self._add_dyn_feat(df_dynamic_features,
                                                         df_feat=self.specific_dynamic_features[dataset],
                                                         min_week=min_week,
                                                         cutoff=self.cutoff,
                                                         future_weeks=self.prediction_length)
                logger.debug(f"Added specific dynamic feature {dataset} to `df_dynamic_features`")

        return df_target, df_static_features, df_dynamic_features

    def deepar_formatting(self, df_target, df_static_features, df_dynamic_data):
        """Method formatting datasets to DeepAR-specific scheme.

        Takes target, static & dynamic features datasets, label-encodes & puts data in jsonline-compliant format.

        Args:
            df_target (pd.DataFrame): Temporal & model-specific dataset including target values up to instance cutoff
            df_static_features (pd.DataFrame): Model-specific dataset with static labels
            df_dynamic_features (pd.DataFrame): Temporal & model-specific dataset with dynamic values

        Returns:
            train_jsonline (str): jsonline DeepAR-compliant string for training (limited to instance cutoff)
            predict_jsonline (str): jsonline DeepAR-compliant string for inference (limited to instance cutoff
                for target, and with future projection for dynamic features)
            """

        # Label Encode Categorical features (also limiting df_static_data to avoid missing cat label error)
        if df_static_features is not None:
            df_static_features = df_static_features.merge(df_target[['model_id']].drop_duplicates(), on=['model_id'])
            for c in self.static_features.keys():
                le = LabelEncoder()
                df_static_features[c] = le.fit_transform(df_static_features[c])
            df_static_features['cat'] = df_static_features[self.static_features.keys()].values.tolist()
            logger.debug("Static features label-encoded.")

        # Building df_predict
        # Adding prediction weeks necessary for dynamic features in df_predict
        df_predict = self._add_future_weeks(df_target)
        if df_dynamic_data is not None:
            df_predict = df_predict.merge(df_dynamic_data, on=['model_id', 'week_id'], how='left')
            # Limiting df_dynamic_data to ensure that unwanted week_ids are not in the dataset
            df_dynamic_data = df_dynamic_data.merge(df_predict[['model_id', 'week_id']].drop_duplicates(),
                                                    on=['model_id', 'week_id'],
                                                    how='inner')
        df_predict.sort_values(by=['model_id', 'week_id'], ascending=True, inplace=True)
        # Building data `start` & `target`
        df_predict = df_predict.groupby(by=['model_id'], sort=False)\
            .agg(start=('week_id', lambda x: ut.week_id_to_date(x.min()).strftime('%Y-%m-%d %H:%M:%S')),
                 target=('sales_quantity', lambda x: list(x.dropna())))
        # Adding categorical features
        if df_static_features is not None:
            df_predict = df_predict.merge(df_static_features[['model_id', 'cat']],
                                          left_index=True,
                                          right_on='model_id'
                                          ).set_index('model_id')
            logger.debug("Added categorical features to `df_predict`")

        # Identifying final list of dynamic features
        if df_dynamic_data is not None:
            dynamic_features = [feat for feat in self.df_dynamic_data.columns if feat not in ['model_id', 'week_id']]
            # Concatenating dynamic features in list format
            df_dynamic_data_predict = df_dynamic_data.sort_values(by=['model_id', 'week_id'], ascending=True)\
                .groupby(by=['model_id'], sort=False)\
                .agg({feat: list for feat in dynamic_features})
            df_dynamic_data_predict['dynamic_feat'] = df_dynamic_data_predict.values.tolist()
            # Adding dynamic features
            df_predict = df_predict.merge(df_dynamic_data_predict[['dynamic_feat']],
                                          left_index=True,
                                          right_index=True,
                                          how='left'
                                          )
            logger.debug("Added dynamic features to `df_predict`")
        df_predict.reset_index(inplace=True)

        # Building df_train
        # Limiting dataset to avoid any future data
        df_train = df_target[df_target['week_id'] < self.cutoff]
        if df_dynamic_data is not None:
            df_dynamic_data_train = df_dynamic_data.merge(df_train[['model_id', 'week_id']].drop_duplicates(),
                                                          on=['model_id', 'week_id'],
                                                          how='inner'
                                                          )
        # Building data `start` & `target`
        df_train.sort_values(by=['model_id', 'week_id'], ascending=True, inplace=True)
        df_train = df_train.groupby(by=['model_id'], sort=False)\
            .agg(start=('week_id', lambda x: ut.week_id_to_date(x.min()).strftime('%Y-%m-%d %H:%M:%S')),
                 target=('sales_quantity', lambda x: list(x.dropna())))
        # Adding categorical features
        if df_static_features is not None:
            df_train = df_train.merge(df_static_features[['model_id', 'cat']],
                                      left_index=True,
                                      right_on='model_id').set_index('model_id')
            logger.debug("Added static features to `df_train`")
        # Concatenating dynamic features in list format
        if df_dynamic_data is not None:
            df_dynamic_data_train = df_dynamic_data_train.sort_values(by=['model_id', 'week_id'], ascending=True)\
                .groupby(by=['model_id'], sort=False)\
                .agg({feat: list for feat in dynamic_features})
            df_dynamic_data_train['dynamic_feat'] = df_dynamic_data_train.values.tolist()
            # Adding dynamic features
            df_train = df_train.merge(df_dynamic_data_train[['dynamic_feat']],
                                      left_index=True,
                                      right_index=True,
                                      how='left'
                                      )
            logger.debug("Added dynamic features to `df_train`")
        # Shuffling df_train
        df_train = df_train.sample(frac=1)
        df_train.reset_index(inplace=True)

        # Converting to jsonline
        train_jsonline = df_train.to_json(orient='records', lines=True)
        predict_jsonline = df_predict.to_json(orient='records', lines=True)
        logger.debug("Jsonline files created")

        # Checking jsonline datasets
        logger.info("Checking train jsonline file...")
        self.check_json_line(train_jsonline)
        logger.info("Checking predict jsonline file...")
        self.check_json_line(predict_jsonline, future_proj_len=self.prediction_length)
        logger.debug("All checks on jsonline files passed")

        return train_jsonline, predict_jsonline

    def import_base_data(self):
        """Helper for potential data import if S3 URI was provided. Also ensure datetime format for 'date' field.
        """
        for dataset in self.base_data.keys():
            if isinstance(self.base_data[dataset], str):
                logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                s3_uri = self.base_data[dataset]
                bucket, path = ut.from_uri(s3_uri)
                self.base_data[dataset] = ut.read_multipart_parquet_s3(bucket, path)
                logger.debug(f"Base data {dataset} imported from S3.")
            if dataset == 'model_week_sales':
                self.base_data[dataset].loc[:, 'date'] = pd.to_datetime(self.base_data[dataset].loc[:, 'date'])

    def import_static_features(self):
        """Helper for potential data import for static features if S3 URI was provided.
        """
        for dataset in self.static_features.keys():
            if isinstance(self.static_features[dataset], str):
                logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                s3_uri = self.static_features[dataset]
                bucket, path = ut.from_uri(s3_uri)
                self.static_features[dataset] = ut.read_multipart_parquet_s3(bucket, path)
                logger.debug(f"Static feature {dataset} imported from S3.")
            assert len(set(self.static_features[dataset].columns) - set(['model_id'])) == 1, \
                f"Static feature dataset {dataset} must contains only one column, aside 'model_id' & 'week_id'"

    def import_global_dynamic_features(self):
        """Helper for potential data import for global dynamic features if S3 URI was provided.

        Also provides a forward fill function if requested.
        Checks if dynamic features datasets contain only one features each
        """
        for dataset in self.global_dynamic_features.keys():
            if isinstance(self.global_dynamic_features[dataset], str):
                logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                bucket, path = ut.from_uri(self.global_dynamic_features[dataset])
                self.global_dynamic_features[dataset] = ut.read_multipart_parquet_s3(bucket, path)
                logger.debug(f"Global dynamic feature {dataset} imported from S3.")
            if self.global_dynamic_features_projection[dataset] == 'ffill':
                self.global_dynamic_features[dataset] = features_forward_fill(df=self.global_dynamic_features[dataset],
                                                                              cutoff=self.cutoff,
                                                                              projection_length=self.prediction_length)
            assert len(set(self.global_dynamic_features[dataset].columns) - set(['week_id'])) == 1, \
                f"Global dynamic feature dataset {dataset} must contains only one column, aside 'model_id' & 'week_id'"

    def import_specific_dynamic_features(self):
        """Helper for potential data import for specific dynamic features if S3 URI was provided.

        Also provides a forward fill function if requested.
        Checks if dynamic features datasets contain only one features each
        """
        for dataset in self.specific_dynamic_features.keys():
            if isinstance(self.specific_dynamic_features[dataset], str):
                logger.info(f"Dataset {dataset} not passed to data handler, importing data from S3...")
                bucket, path = ut.from_uri(self.specific_dynamic_features[dataset])
                self.specific_dynamic_features[dataset] = ut.read_multipart_parquet_s3(bucket, path)
                logger.debug(f"Global dynamic feature {dataset} imported from S3.")
            if self.specific_dynamic_features_projection[dataset] == 'ffill':
                self.specific_dynamic_features[dataset] = \
                    features_forward_fill(df=self.specific_dynamic_features[dataset],
                                          cutoff=self.cutoff,
                                          projection_length=self.prediction_length
                                          )
            assert len(set(self.specific_dynamic_features[dataset].columns) - set(['week_id', 'model_id'])) == 1, \
                f"Static feature dataset {dataset} must contains only one column, aside 'model_id' & 'week_id'"

    def _add_static_feat(self, df_static_features, df_feat):
        """Adds unitary static feature to the concatenated static features dataset

        Args:
            df_static_features (pd.DataFrame): Concatenated dataframe for static features. Can be empty except for
                'model_id'. The declared models will be exactly the models exposed to the modeling step.
            df_feat (pd.DataFrame): Static features dataset. Must include column 'model_id' and one column with the
                associated static feature. The name of this column will be the name of the feature.

        Returns:
            df_with_new_feat (pd.DataFrame): Updted concatenated dataframe for static features.
        """

        feature_name = ', '.join((set(df_feat.columns) - set(['model_id', 'week_id'])))
        assert 'model_id' in df_feat.columns, f"Columns `model_id` is missing from static dataset {feature_name}"

        # Checking if `model_id` and `week_id` lists match in both df_dynamic_data and df_feat
        assert len(set(df_static_features['model_id'].unique()) - set(df_feat['model_id'].unique())) == 0,\
            "Mismatch in model_id list between the dynamic feature and the sales dataset"

        df_with_new_feat = pd.merge(df_static_features, df_feat, on=['model_id'])

        return df_with_new_feat

    def _add_dyn_feat(self, df_dynamic_data, df_feat, min_week, cutoff, future_weeks, week_column='week_id'):
        """Adds unitary dynamic feature to the concatenated static features dataset.

        If the feature is global (not model-specific), it will explode the feature for each model in `df_dynamic_data`

        Args:
            df_dynamnic_data (pd.DataFrame): Concatenated dataframe for dynamic features. Can be empty except for
                'model_id' and 'weekd_id'. The declared models and weeks will be exactly the models exposed to
                the modeling step.
            df_feat (pd.DataFrame): Static features dataset. Must include column 'model_id' and one column with the
                associated static feature. The name of this column will be the name of the feature.
            min_week (int): Minimum week to check in the DataFrame (YYYYWW ISO Format)
            cutoff (int): instance cutoff
            future_weeks (int): number of weeks in the future to infer on
            week_column (str): Column name for weeks

        Returns:
            df_with_new_feat (pd.DataFrame): Updted concatenated dataframe for dynamic features.
        """

        # Checks
        check_weeks_df(df_feat, min_week, cutoff, future_weeks, week_column=week_column)

        if 'model_id' not in df_feat.columns:
            # assert models is not None, "If `df_feat` is a global dynamic feature, you must provide `models` with
            # a pd.DataFrame of expected models with 'model_id' column"
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
        """Provides a model-specific dataframe with future weeks to infer on.

        Args:
            df (pd.DataFrame): dataframe including columns 'week_id' & 'model_id'. Can contains other columns.

        Returns:
            df (pd.DataFrame): updated dataframe appended with future weeks (exploded on 'model_id'). Other columns
                are filled with np.nan
        """

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
        """Checks DeepAR jsonline conformity

        Args:
            jsonline (str): Jsonline-compliant string
            future_proj_len (int): future weeks to infer on
        """

        df = pd.read_json(jsonline, orient='records', lines=True)

        # Test if target >= rec_length
        df['target_len'] = df.apply(lambda x: len(x['target']), axis=1)
        test = df['target_len'] >= self.rec_length
        assert all(test.values), 'Some models have a `target` less than `rec_length`'
        logger.debug("All target time series have a length longer than `rec_length`")

        # Test if target length is right
        df['target_len_test'] = \
            df.apply(lambda x: ut.date_to_week_id(pd.to_datetime(x['start']) + pd.Timedelta(x['target_len'], 'W')
                                                  ) == self.cutoff, axis=1
                     )
        assert all(df['target_len_test'].values), \
            "Some models have a 'target' length which doesn't match with the 'start' date"
        logger.debug("All target time series have a length matching with start_date and cutoff")

        if 'cat' in df.columns:
            # Test if right number of categorical features
            df['cat_feat_nb'] = df.apply(lambda x: len(x['cat']), axis=1)
            test = df['cat_feat_nb'] == len(set(self.df_static_data.columns) - set(['model_id']))
            assert all(test.values), "Some models don't have the right number of categorical features"
            logger.debug(f"All models have {len(set(self.df_static_data.columns) - set(['model_id']))} "
                         f"categorical features")

        if 'dynamic_feat' in df.columns:
            nb_dyn_feat = len(set(self.df_dynamic_data.columns) - set(['model_id', 'week_id']))

            # Test if right number of dynamic features
            df['dyn_feat_nb'] = df.apply(lambda x: len(x['dynamic_feat']), axis=1)
            test = df[['dyn_feat_nb']] == nb_dyn_feat
            assert all(test.values), "Some models don't have the right number of dynamic features"
            logger.debug(f"All models have {nb_dyn_feat} dynamic features")

            # Test if right length of dynamic features
            dynamic_features_names = list(self.df_dynamic_data.columns)
            dynamic_features_names.remove('week_id')
            dynamic_features_names.remove('model_id')
            for i, feat in enumerate(dynamic_features_names):
                df[f'dyn_feat_{feat}_len'] = df.apply(lambda x: len(x['dynamic_feat'][i]), axis=1)
                df[f'dyn_feat_{feat}_len_test'] = df[f'dyn_feat_{feat}_len'] == df['target_len'] + future_proj_len

            for feat in dynamic_features_names:
                assert df[df[f'dyn_feat_{feat}_len_test']].shape[0] == df.shape[0], \
                    f"Some models don't have the right dynamic feature length for feature {feat}"
                logger.debug(f"All models have the right dynamic features {feat} length")
