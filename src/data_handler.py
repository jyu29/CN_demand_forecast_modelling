import os
import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from src.utils import (week_id_to_date, date_to_week_id, from_uri, read_yml, 
                       write_str_to_file_on_s3, write_df_to_parquet_on_s3)
from src.refining_specific_functions import (pad_to_cutoff, cold_start_rec, initialize_df_dynamic_features,
                                             is_rec_feature_processing, features_forward_fill, 
                                             apply_first_lockdown_patch)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

SUPPORTED_ALGORITHMS = ['deepar', 'arima']
CONFIG_PATH = 'config'


def import_refining_config(environment: str,
                           algorithm: str,
                           cutoff: int,
                           train_path: str,
                           predict_path: str
                           ) -> dict:
    """Handler to import specific refining configuration from YML file

    Args:
        environment (str): Set of parameters on which to load the parameters
        algorithm (str): Algorithm name
        cutoff (int): Cutoff week in format YYYYWW (ISO 8601)
        train_path (str): Path of training data
        predict_path (str): Path of predict data
        df_jobs (pd.DataFrame): helper to ensure Sagemaker tracking of training & inference, and associated
            files paths.

    Returns:
        A dictionary with all parameters for specific refining process
    """
    assert isinstance(environment, str)
    # the list of algorithms currently supported by the refining
    assert algorithm in SUPPORTED_ALGORITHMS, \
    f"Algorithm {algorithm} not in list of supported algorithms {SUPPORTED_ALGORITHMS}"
    assert isinstance(cutoff, int)
    assert isinstance(train_path, str)
    assert isinstance(predict_path, str)

    params_full_path = os.path.join(CONFIG_PATH, f"{environment}.yml")
    params = ut.read_yml(params_full_path)

    refining_params = {
        'algorithm': algorithm,
        'cutoff': cutoff,
        'patch_first_lockdown': params['refining_specific_parameters']['patch_first_lockdown'],
        'rec_length': params['refining_specific_parameters']['rec_length'],
        'rec_cold_start': params['refining_specific_parameters']['rec_cold_start'],
        'rec_cold_start_group': params['refining_specific_parameters']['rec_cold_start_group'],
        'prediction_length': params['modeling_parameters']['algorithm'][algorithm]['hyperparameters']['prediction_length'],
        'context_length': params['modeling_parameters']['algorithm'][algorithm]['hyperparameters']['context_length'],
        'output_paths': {'train_path': train_path, 'predict_path': predict_path}
    }
    return refining_params


class DataHandler:
    """
    Data Handler from refined data global to feature engineering for
    the demand Forecast project.
    """
    
    def __init__(self,
                 algorithm: str,
                 cutoff: int,
                 patch_first_lockdown: bool,
                 rec_length: int,
                 rec_cold_start: bool,
                 rec_cold_start_group: list,
                 prediction_length: int,
                 context_length: int,
                 output_paths: dict,
                 base_data: dict,
                 static_features: dict = None,
                 global_dynamic_features: dict = None,
                 specific_dynamic_features: dict = None
                 ):
        """Instanciation of data handler.

        The Data Handler takes care of Data Refining for specific aspects forecast algorithms,
        formats the data to the expected format, and save the results in AWS S3.

        Args:
            algorithm (str): Algorithm name
            cutoff (int): Cutoff week in format YYYYWW (ISO 8601)
            patch_first_lockdown (bool): if true, replace the actual sales of the first lockdown with a 
                reconstructed version
            rec_length (int): Minimum weeks expected in each input time series
            rec_cold_start (bool): if true, apply a cold-start reconstruction
            rec_cold_start_group (list): for the cold start reconstruction, columns to use in model_week_tree
            prediction_length (int): Number of forecasted weeks in the future
            context_length (int): Number of sales weeks used as context for forecasting
            output_paths (dict): Dictionnary of refining output S3 uri paths for the train & predict files
            base_data (dict): Dictionnary of pd.DataFrame for base (essential) data
            static_features(dict): Optionnal dictionnary of pd.DataFrame for static data
            global_dynamic_features(dict): Optionnal dictionnary of pd.DataFrame for dynamic data
            specific_dynamic_features(dict): Optionnal dictionnary of pd.DataFrame for time-series
                specific data
        """
        
        self.algorithm = algorithm
        self.cutoff = cutoff
        self.patch_first_lockdown = patch_first_lockdown
        self.rec_length = rec_length
        self.rec_cold_start = rec_cold_start
        self.rec_cold_start_group = rec_cold_start_group
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.output_paths = output_paths
        
        # Statistical algorithms ignore static features
        if algorithm == 'arima':
            static_features=None
            
        # Base data init
        if patch_first_lockdown:
            assert 'imputed_sales_lockdown_1' in base_data.keys(), \
            "Patching first lockdown requested, but imputation dataset not provided in base_data"   
        for key, df in base_data.items():
            assert isinstance(df, pd.DataFrame), f"Value of `{key}` must be a pd.DataFrame"
        self.base_data = base_data
        
        # Static data init
        if static_features:
            for key, df in static_features.items():
                assert isinstance(df, pd.DataFrame), f"Value of `{key}` must be a pd.DataFrame"
        self.static_features = static_features
        
        # Global dynamic data init
        if global_dynamic_features:
            self.global_dynamic_features = {}
            self.global_dynamic_features_projection = {}
            for key, sub_dict in global_dynamic_features.items():
                assert (isinstance(sub_dict, dict)) & (any(k in sub_dict.keys() for k in ['dataset', 'projection'])), \
                f"Value of `{key}` must be a dict with keys 'dataset' & 'projection'"
                assert isinstance(sub_dict['dataset'], pd.DataFrame), \
                f"Value `dataset` of `{key}` must be a pd.DataFrame"
                assert sub_dict['projection'] in ['as_provided', 'ffill'], \
                f"Value `projection` of `{key}` only handles `as_provided` or `ffill`"
                self.global_dynamic_features[key] = sub_dict['dataset']
                self.global_dynamic_features_projection[key] = sub_dict['projection']
        else:
            self.global_dynamic_features = None
            self.global_dynamic_features_projection = None

        # Specific dynamic data init
        if specific_dynamic_features:
            self.specific_dynamic_features = {}
            self.specific_dynamic_features_projection = {}
            for key, sub_dict in specific_dynamic_features.items():
                assert (isinstance(sub_dict, dict)) & (any(k in sub_dict.keys() for k in ['dataset', 'projection'])), \
                f"Value of `{key}` must be a dict with keys 'dataset' & 'projection'"
                assert isinstance(sub_dict['dataset'], pd.DataFrame), \
                f"Value `dataset` of `{key}` must be a pd.DataFrame"
                assert sub_dict['projection'] in ['as_provided', 'ffill'], \
                f"Value `projection` of `{key}` only handles `as_provided` or `ffill`"
                self.specific_dynamic_features[key] = sub_dict['dataset']
                self.specific_dynamic_features_projection[key] = sub_dict['projection']
        else:
            self.specific_dynamic_features = None
            self.specific_dynamic_features_projection = None

        # Loggs
        logger.info(f"Data refining specific for Demand Forecast initialized for algorithm {self.algorithm} "
                    f"and cutoff {self.cutoff}")
        
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

        Executes all steps for data refining specific to demand forecasting:
        * Input data processing (including check & potential forward fill)
        * Data refining specific to demand forecasting, including:
          * Dataset limitation to instance cutoff & MRP active products
          * Pad to cutoff for missing data (models for which sales stopped before the instance cutoff)
          * Optional lockdown and cold start reconstruction (depending on parameters)
          * Static & dynamic features concatenation
        * Formatting datasets to the algorithm expected scheme
        * Upload to S3
        """

        # Process input data
        self.process_input_data()

        # Refining specific
        logger.info("Starting Data Refining...")
        self.df_target, self.df_static_features, self.df_dynamic_features = self.refining_specific()

        # Formatting
        if self.algorithm == 'deepar':
            logger.info("Starting DeepAR formatting...")
            train_jsonline, predict_jsonline = self.deepar_formatting(self.df_target,
                                                                      self.df_static_features,
                                                                      self.df_dynamic_features
                                                                      )
            # Saving jsonline files on S3
            train_bucket, train_path = from_uri(self.output_paths['train_path'])
            predict_bucket, predict_path = from_uri(self.output_paths['predict_path'])
            
            write_str_to_file_on_s3(train_jsonline, train_bucket, f"{train_path}")
            logger.info(f"Train jsonline file saved at {self.output_paths['train_path']}")
            
            write_str_to_file_on_s3(predict_jsonline, predict_bucket, f"{predict_path}")
            logger.info(f"Predict jsonline file saved at {self.output_paths['predict_path']}")
        
        if self.algorithm == 'arima':
            logger.info("Starting ARIMA formatting...")
            df_train = self.arima_formatting(self.df_target,
                                             df_dynamic_features=None # ARIMAX not tested yet
                                             )
            # Saving dataframe on S3
            train_bucket, train_path = from_uri(self.output_paths['train_path'])
            
            write_df_to_parquet_on_s3(df_train, train_bucket, f"{train_path}")
            logger.info(f"Train parquet file saved at {self.output_paths['train_path']}")
            

    def process_input_data(self):
        """Workflow method to process datasets.

        Provides a workflow to process base data (used for cold start reconstruction for example), static features,
        global dynamic features (temporal features not specific to a model), and specific dynamic features (temporal
        features defined at model level)
        """
        
        # Process base data
        for key, df in self.base_data.items():
            assert ('model_id' in df.columns) & ('week_id' in df.columns), \
            f"Base dataset `{key}` must contains columns `model_id` & `week_id`"
            if 'date' in df.columns:
                self.base_data[key].loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'])
        logger.info("Attribute `base_data` processed.")

        # Process static features data
        if self.static_features:
            for key, df in self.static_features.items():
                assert (any(c in df.columns for c in ['model_id', key]) & (df.shape[1] == 2)), \
                f"Static feature dataframe `{key}` must contain only columns `model_id` and `{key}`"
            logger.info("Attribute `static_features` processed")
        else:
            logger.info("No static features specified.")
        
        # Process global dynamic features
        if self.global_dynamic_features:
            for key, df in self.global_dynamic_features.items():
                assert (any(c in df.columns for c in ['week_id', key]) & (df.shape[1] == 2)), \
                f"Global dynamic feature dataframe `{key}` must contain only columns `week_id` and `{key}`"
                if self.global_dynamic_features_projection[key] == 'ffill':
                    self.global_dynamic_features[key] = \
                        features_forward_fill(df=df,
                                              cutoff=self.cutoff,
                                              projection_length=self.prediction_length)
            logger.info("Attribute `global_dynamic_features` processed")
        else:
            logger.info("No global dynamic features specified.")

        # Process specific dynamic features
        if self.specific_dynamic_features:
            for key, df in self.specific_dynamic_features.items():
                assert (any(c in df.columns for c in ['model_id', 'week_id', key]) & (df.shape[1] == 3)), \
                f"Specific dynamic feature dataframe `{key}` must contain only columns `model_id`, `week_id` and `{key}`"
                if self.specific_dynamic_features_projection[key] == 'ffill':
                    self.specific_dynamic_features[key] = \
                        features_forward_fill(df=df,
                                              cutoff=self.cutoff,
                                              projection_length=self.prediction_length)
            logger.info("Attribute `specific_dynamic_features` processed")
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
          * Static & dynamic features concatenation

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

        # Creating df_target
        df_target = df_sales[['model_id', 'week_id', 'sales_quantity']]

        # Initializing and filling df_static_features
        if self.static_features:
            df_static_features = df_target[['model_id']].drop_duplicates().copy()
            for feature_name, df_new_feat in self.static_features.items():
                df_static_features = self._add_feature(df_features=df_static_features,
                                                       df_new_feat=df_new_feat,
                                                       feature_name=feature_name)
                logger.debug(f"Added static feature `{feature_name}` to `df_static_features`")

        # Initializing df_dynamic_features
        if any([self.rec_cold_start, self.global_dynamic_features, self.specific_dynamic_features]):
            df_dynamic_features = initialize_df_dynamic_features(df=df_target,
                                                                 cutoff=self.cutoff,
                                                                 prediction_length=self.prediction_length
                                                                 )

        # Building is_rec specific dynamic feature and adding it to df_dynamic_features
        if self.rec_cold_start:
            df_is_rec = is_rec_feature_processing(df_sales, self.cutoff, self.prediction_length)
            df_dynamic_features = self._add_feature(df_features=df_dynamic_features,
                                                    df_new_feat=df_is_rec,
                                                    feature_name='is_rec')
            logger.debug("Cold start reconstruction requested. Added global dynamic feature `is_rec` to `df_dynamic_features`.")

        # Adding provided global dynamic features
        if self.global_dynamic_features:
            for feature_name, df_new_feat in self.global_dynamic_features.items():
                df_dynamic_features = self._add_feature(df_features=df_dynamic_features,
                                                        df_new_feat=df_new_feat,
                                                        feature_name=feature_name)
                logger.debug(f"Added global dynamic feature `{feature_name}` to `df_dynamic_features`")

        # Adding provided specific dynamic features
        if self.specific_dynamic_features:
            for feature_name, df_new_feat in self.specific_dynamic_features.items():
                df_dynamic_features = self._add_feature(df_features=df_dynamic_features,
                                                        df_new_feat=df_new_feat,
                                                        feature_name=feature_name)
                logger.debug(f"Added specific dynamic feature {feature_name} to `df_dynamic_features`")

        return df_target, df_static_features, df_dynamic_features

    def deepar_formatting(self, df_target, df_static_features, df_dynamic_features):
        """Method formatting datasets to Sagemaker DeepAR-specific scheme.

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
        
        # Init df_train
        df_train = df_target.copy()
            
        # Formatting & building `start` & `target` columns
        df_train = df_train \
            .sort_values(['model_id', 'week_id']) \
            .groupby('model_id') \
            .agg(start=('week_id', lambda x: week_id_to_date(x.min()).strftime('%Y-%m-%d %H:%M:%S')),
                 target=('sales_quantity', lambda x: list(x)))
        logger.debug("Built start & target columns from `df_train`")
        
        # Encoding the labels & adding static features
        if df_static_features is not None:
            df_static_features_train = df_static_features.copy()
            for c in self.static_features.keys():
                le = LabelEncoder()
                df_static_features_train[c] = le.fit_transform(df_static_features_train[c])
            logger.debug("Static features label-encoded.")
            
            df_static_features_train['cat'] = df_static_features_train[self.static_features.keys()].values.tolist()
            df_static_features_train.set_index('model_id', inplace=True)
            
            df_train = df_train.merge(df_static_features_train[['cat']],
                                      left_index=True,
                                      right_index=True,
                                      how='left')
            logger.debug("Added static features to `df_train`")
        
        # Building df_predict from df_train
        df_predict = df_train.copy()
        
        # Formatting & adding dynamic features
        if df_dynamic_features is not None:
            
            l_dynamic_features = [feat for feat in df_dynamic_features.columns if feat not in ['model_id', 'week_id']]
            
            # Exluding feat projection for train
            df_dynamic_features_train = df_dynamic_features \
                .loc[df_dynamic_features['week_id'] < self.cutoff] \
                .sort_values(['model_id', 'week_id']) \
                .groupby('model_id') \
                .agg({feat: list for feat in l_dynamic_features})
            
            # Including feat projection for predict
            df_dynamic_features_predict = df_dynamic_features \
                .sort_values(['model_id', 'week_id']) \
                .groupby('model_id') \
                .agg({feat: list for feat in l_dynamic_features})

            df_dynamic_features_train['dynamic_feat'] = df_dynamic_features_train.values.tolist()
            df_dynamic_features_predict['dynamic_feat'] = df_dynamic_features_predict.values.tolist()

            df_train = df_train.merge(df_dynamic_features_train[['dynamic_feat']], 
                                      left_index=True, 
                                      right_index=True,
                                      how='left')
            
            df_predict = df_predict.merge(df_dynamic_features_predict[['dynamic_feat']], 
                                          left_index=True, 
                                          right_index=True, 
                                          how='left')
            
            logger.debug("Added dynamic features to `df_train` & `df_predict`")
            
        # Shuffling df_train
        df_train = df_train.sample(frac=1)
        
        # Resetting index to retrieve model_id column
        df_train.reset_index(inplace=True)
        df_predict.reset_index(inplace=True)
        
        # Converting to jsonline
        train_jsonline = df_train.to_json(orient='records', lines=True)
        predict_jsonline = df_predict.to_json(orient='records', lines=True)
        logger.debug("Jsonline files created")

        # Checking jsonline datasets
        logger.info("Checking output jsonline files...")
        self.check_deepar_json_line(train_jsonline)
        self.check_deepar_json_line(predict_jsonline, future_proj_len=self.prediction_length)
        logger.debug("All checks on jsonline files passed")

        return train_jsonline, predict_jsonline
    
    def arima_formatting(self, df_target, df_dynamic_features):
        """Method formatting datasets to ARIMA-specific scheme.

        Args:
            df_target (pd.DataFrame): Temporal & model-specific dataset including target values up to instance cutoff
            df_dynamic_features (pd.DataFrame): Temporal & model-specific dataset with dynamic values

        Returns:
            df_train (pd.DataFrame): ARIMA-compliant pd.DataFrame for training & inference
            """

        df_train = df_target.copy()
        
        # Optionnaly add dynamic data
        if df_dynamic_features is not None:
            
            df_train = df_train.merge(df_dynamic_features, 
                                      on=['model_id', 'week_id'], 
                                      how='outer')
            logger.debug("Added dynamic features to `df_train`")
            
        df_train.sort_values(['model_id', 'week_id'], inplace=True)
        
        return df_train

    def _add_feature(self, df_features, df_new_feat, feature_name):
        """Adds unitary new feature to the concatenated df_features

        Args:
            df_features (pd.DataFrame): Concatenated features dataframe.
            df_new_feat (pd.DataFrame): The new feature to be added to 'df_features'.
            feature_name (str): The feature name (key from 'static_features' input dict). Used for debugging purposes.
        Returns:
            df_with_new_feat (pd.DataFrame): Updated concatenated features dataframe.
        """

        df_with_new_feat = pd.merge(df_features, df_new_feat, how='left')
        
        assert df_with_new_feat.notnull().values.any(), \
        f"Missing information for feature {feature_name} (missing model_id and/or week_id, according to feature type)"
            
        return df_with_new_feat

    def check_deepar_json_line(self, jsonline, future_proj_len=0):
        """Checks DeepAR jsonline conformity
    
        Args:
            jsonline (str): Jsonline-compliant string
            future_proj_len (int): future weeks to infer on
        """
    
        df = pd.read_json(jsonline, orient='records', lines=True)
        
        # Calculate target length
        df['target_len'] = df.apply(lambda x: len(x['target']), axis=1)
        
        # Test if target length is consistent
        df['is_len_consistent'] = \
            df.apply(lambda x: date_to_week_id(pd.to_datetime(x['start']) + pd.Timedelta(x['target_len'], 'W')
                                              ) == self.cutoff, axis=1
                    )
        assert all(df['is_len_consistent']), \
        "Some models have a 'target' length which doesn't match with the 'start' date"
        logger.debug("All target time series have a length matching with start_date and cutoff")
        
        # Test if len(target) >= prediction_length + context_length
        df['is_len_long_enough'] = df['target_len'] >= self.prediction_length + self.context_length
        assert all(df['is_len_long_enough']), \
        'Some models have a `target` less than `prediction_length` + `context_length`'
        logger.debug("All target time series are long enough")
    
        if 'cat' in df.columns:
            # Test if right number of static (cat) features
            nb_cat_feat_expected = len(set(self.df_static_features.columns) - set(['model_id']))
            df['nb_cat_feat'] = df.apply(lambda x: len(x['cat']), axis=1)
            df['is_nb_cat_ok'] = df['nb_cat_feat'] == nb_cat_feat_expected
            assert all(df['is_nb_cat_ok']), "Some models don't have the right number of static features"
            logger.debug(f"All target time series have the correct number of static features")
    
        if 'dynamic_feat' in df.columns:
            # Test if right number of dynamic features
            nb_dyn_feat_expected = 1
            df['nb_dyn_feat'] = df.apply(lambda x: np.array(x['dynamic_feat']).shape[0], axis=1)
            df['is_nb_dyn_ok'] = df['nb_dyn_feat'] == nb_dyn_feat_expected
            assert all(df['is_nb_dyn_ok']), "Some models don't have the right number of dynamic features"
            logger.debug(f"All target time series have the correct number of dynamic features")
    
            # Test if right length of dynamic features
            len_dyn_feat_expected = df['target_len'] + future_proj_len
            df['len_dyn_feat'] = df.apply(lambda x: np.array(x['dynamic_feat']).shape[1], axis=1)
            df['is_len_dyn_ok'] = df['len_dyn_feat'] == len_dyn_feat_expected
            assert all(df['is_len_dyn_ok']), "Some models don't have the right dynamic feature length"
            logger.debug(f"All target time series have the right number of dynamic features length")
