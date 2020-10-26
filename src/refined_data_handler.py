import pandas as pd
import numpy as np
import json
import random
import time
import datetime
import src.utils as ut
from sklearn.preprocessing import LabelEncoder

###############################################################################
########################### Dynamic Feature Handler ###########################
###############################################################################


class dynamic_feature_handler:
    """ Handler for dynamic features. Will check if `model` is in columns ; if so, the handler will be at model level.
        If not, it will be applied for all models.
    """

    def __init__(self, bucket, cutoff, feat_name, feat_path, hist_rec_method, proj_method, proj_length):

        self.bucket = bucket
        self.cutoff = cutoff
        self.feat_name = feat_name
        self.feat_path = feat_path
        self.hist_rec_method = hist_rec_method
        self.proj_method = proj_method
        self.proj_length = proj_length

    def import_input_datasets(self):

        # Read CSV
        df_feat = ut.read_csv_s3(self.bucket, self.feat_path, sep="|")

        # Check columns
        assert self.feat_name in df_feat.columns
        assert 'week_id' in df_feat.columns
        if 'model' in df_feat.columns:
            print(f'    {self.feat_name} is defined at model level')
            self._model_level = True
        else:
            self._model_level = False

        # Save attributes
        self.df_feat = df_feat[df_feat['week_id'] < self.cutoff].copy()
        self.df_future_raw = df_feat[df_feat['week_id'] >= self.cutoff].copy()

    def history_reconstruction(self):

        if self.hist_rec_method == 'fill':
            if self._model_level:
                self.df_hist = self._hist_rec_fill(self.df_feat)
                print(f"    History reconstruction `{self.hist_rec_method}` was applied")
            else:
                raise NameError(
                    f"History reconstruction algorithm `{self.hist_rec_method}` only exists at model-level for the moment")
        elif self.hist_rec_method is None:
            self.df_hist = self.df_feat
            print(f"    No history reconstruction was applied")
        else:
            raise NameError(f"History reconstruction algorithm `{self.hist_rec_method}` doesn't exist")

    def future_projection(self):
        assert hasattr(self, 'df_hist'), "History reconstruction must be done before future projection. " + \
            "If you don't want to do history reconstruction, please init `hist_rec_method` to None"

        if self.proj_method == 'last_value':
            self.df_future = self._fut_proj_last_value(self.df_hist)

        elif self.proj_method == 'seasonal_naive':
            self.df_future = self._fut_proj_seasonal_naive(self.df_hist)

        elif self.proj_method == 'as_provided':
            self._check_ts_df_format(self.df_future_raw, self.proj_length)
            max_week = ut.date_to_week_id(ut.week_id_to_date(self.cutoff) + datetime.timedelta(weeks=self.proj_length))
            self.df_future_raw = self.df_future_raw[self.df_future_raw['week_id'] < max_week]
            self.df_future = pd.concat([self.df_hist, self.df_future_raw], ignore_index=True)
        else:
            raise NameError(f"Future projection algorithm `{self.proj_method}` doesn't exist")

        if self._model_level:
            self.df_future.sort_values(['model', 'week_id'], ascending=True, inplace=True)
        else:
            self.df_future.sort_values(['week_id'], ascending=True, inplace=True)

    def _hist_rec_fill(self, df):

        # Create a complete TS dataframe
        all_model = df['model'].sort_values().unique()
        all_week = df['week_id'].sort_values().unique()
        w, m = pd.core.reshape.util.cartesian_product([all_week, all_model])
        complete_ts = pd.DataFrame({'model': m, 'week_id': w})

        # Checking that we have enough weeks in the past
        assert len(w) > self.min_ts_len, f"Dataframe for dynamic feature {self.feat_name} doesn't have enough weeks"

        # Merging with feature values
        complete_ts = complete_ts.merge(df[['model', 'week_id', self.feat_name]], on=['model', 'week_id'], how='left')

        # Sorting for the ffill & bfill
        complete_ts.sort_values(['model', 'week_id'], inplace=True)

        # Ffill first for intermediate values, bfill then for missing values at the beginning
        complete_ts = complete_ts.groupby(['model']).apply(lambda v: v.ffill().bfill())

        return complete_ts

    def _fut_proj_last_value(self, df):

        all_date = pd.Series(pd.date_range(start=ut.week_id_to_date(self.cutoff), periods=self.proj_length, freq='W'))
        all_week = ut.date_to_week_id(all_date)

        if self._model_level:
            all_model = df['model'].sort_values().unique()
            w, m = pd.core.reshape.util.cartesian_product([all_week, all_model])
            df_last_values = df[['model', self.feat_name]].groupby('model').last()[self.feat_name]
            df_future = pd.DataFrame({'model': m, 'week_id': w})
            df_future = df_future.merge(df_last_values, on='model', how='inner')
            df = df.append(df_future)
            df = df.sort_values(['model', 'week_id']).reset_index()
        else:
            last_value = df.loc[df['week_id'] == df['week_id'].max(), self.feat_name].values[0]
            df_future = pd.DataFrame({'week_id': all_week, self.feat_name: last_value})
            df = df.append(df_future).sort_values(['week_id'])

        return df

    def _fut_proj_seasonal_naive(self, df):

        assert self.proj_length <= 52, "Can't handle projection more than 1 year in the future for now"

        df_future = pd.DataFrame()
        all_date = pd.Series(pd.date_range(start=ut.week_id_to_date(self.cutoff), periods=self.proj_length, freq='W'))
        all_week = ut.date_to_week_id(all_date)
        
        if self._model_level:
            df.sort_values(['model', 'week_id'], inplace=True)
            for m, mini_df in df.groupby('model'):
                value = mini_df[self.feat_name].to_list()[-self.proj_length:]
                df_future = df_future.append(pd.DataFrame({'model': m,
                                                           'week_id': weekrange,
                                                           self.feat_name: value
                                                           }))
            df = df.append(df_future)
            df = df.sort_values(['model', 'week_id']).reset_index()
        else:
            df.sort_values(['week_id'], inplace=True)
            value = df[self.feat_name].to_list()[-52:-52+self.proj_length]
            df_future = pd.DataFrame({'week_id': all_week, self.feat_name: value})
            df = df.append(df_future).sort_values(['week_id'])

        return df

    def _check_ts_df_format(self, df, length):
        """This function checks a dataframe on :
        * the number of consecutive weeks
        * if applicable, for each model
        """
        model_level = 'model' in df.columns

        # Building the cartesian product for weeks (and models, if applicable)
        start_date = ut.week_id_to_date(df['week_id'].min())
        end_date = start_date + datetime.timedelta(weeks=length-1)
        week_id_range = [ut.date_to_week_id(i) for i in pd.date_range(start=start_date, end=end_date, freq='W')]

        # Ignoring weeks after end_date
        df = df[df['week_id'] <= max(week_id_range)]

        if model_level:
            model_list = list(df['model'].unique())
            w, m = pd.core.reshape.util.cartesian_product([week_id_range, model_list])
            df_test_all = pd.DataFrame({'model': m, 'week_id': w})
            df_test = df_test_all.merge(df[['model', 'week_id']], on=['model', 'week_id'], how='outer', indicator=True)
            assert df_test[df_test['_merge'] == 'both'].shape[0] == df_test_all.shape[0], \
                'Dataframe has missing weeks for at least one model'
        else:
            # checking length
            assert len(week_id_range) >= length, 'Dataframe time range is less than expected length'
            assert week_id_range == list(df['week_id'].unique()), 'Dataframe has missing weeks'

###############################################################################
########################### Refined Data Handler ##############################
###############################################################################


class refined_data_handler():
    def __init__(self, params):

        assert 'cutoff' in params
        assert 'bucket' in params
        assert 'cat_cols' in params
        assert 'min_ts_len' in params
        assert 'prediction_length' in params
        assert 'clean_data_path' in params
        assert 'run_input_path' in params
        assert 'hist_rec_method' in params
        assert 'dyn_cols' in params

        if params['hist_rec_method'] == 'cluster_avg':
            assert 'cluster_keys' in params
            assert 'patch_covid' in params

        if params['dyn_cols'] != None:
            for c in params['dyn_cols']:
                assert 'hist_rec_method' in params['dyn_cols'][c]
                assert 'proj_method' in params['dyn_cols'][c]
                if params['dyn_cols'][c]['hist_rec_method'] == 'cluster_avg':
                    assert 'cluster_keys' in params['dyn_cols'][c]
                    assert 'patch_covid' in params['dyn_cols'][c]

        self.params = params
        self.cutoff = params['cutoff']
        self.bucket = params['bucket']
        self.cat_cols = params['cat_cols']
        self.min_ts_len = params['min_ts_len']
        self.prediction_length = params['prediction_length']
        self.hist_rec_method = params['hist_rec_method']
        self.dyn_cols = params['dyn_cols']
        self.paths = {'actual_sales': f"{params['clean_data_path']}df_actual_sales.csv",
                      'model_info': f"{params['clean_data_path']}df_model_info.csv",
                      'mrp': f"{params['clean_data_path']}df_mrp.csv",
                      'run_input': params['run_input_path']}

        self._input_data_imported = False

        self.df_train = None
        self.df_predict = None
        self.dyn_train = None
        self.dyn_predict = None

    def import_input_datasets(self):

        # Read CSV
        df_actual_sales = ut.read_csv_s3(self.bucket, self.paths['actual_sales'], parse_dates=['date'], sep='|')
        df_model_info = ut.read_csv_s3(self.bucket, self.paths['model_info'], sep='|')
        df_mrp = ut.read_csv_s3(self.bucket, self.paths['mrp'], sep='|')

        # Filter on cutoff & format
        df_actual_sales = df_actual_sales[df_actual_sales['week_id'] < self.cutoff]
        df_model_info = df_model_info[df_model_info['week_id'] == self.cutoff]
        df_mrp = df_mrp[df_mrp['week_id'] == self.cutoff]

        df_actual_sales.rename(columns={'date': 'ds', 'qty': 'y'}, inplace=True)

        # Save as attributes
        self.df_actual_sales = df_actual_sales
        self.df_model_info = df_model_info
        self.df_mrp = df_mrp
        
        # Import validation
        self._input_data_imported = True

        print("Datasets imported : \n"
              + f"    {self.paths['actual_sales']}\n"
              + f"    {self.paths['model_info']}\n"
              + f"    {self.paths['mrp']}\n")

    def generate_deepar_input_data(self, fs):
        assert self._input_data_imported, "Cutoff data not imported, please use method self.import_input_datasets()"

        # Generating train/predict datasets for target data
        print(f"Generating target dataset for cutoff {self.cutoff}...")
        self._generate_target_data()

        if self.dyn_cols:
            # Generating train/predict datasets for dyn feature data
            print(f"Generating dynamic features datasets for cutoff {self.cutoff}...")
            self._generate_dyn_feature_data()

        print("Generating jsonline file for train dataset...")
        # Merging target with categorical features & dynamic features if exists
        self.df_train = self._merge_target_and_features(self.df_train, self.dyn_train)
        # Shuffling train dataset
        self.df_train = self._shuffle_dataset(self.df_train)
        # Generating jsonline file
        self.train_jsonline = self._df_to_jsonline(self.df_train)

        print("Generating jsonline file for predict dataset...")
        # Merging target with categorical features & dynamic features if exists
        self.df_predict = self._merge_target_and_features(self.df_predict, self.dyn_predict)
        # Generating jsonline file
        self.predict_jsonline = self._df_to_jsonline(self.df_predict)

        # Functional check on jsonlines
        self.check_json_line(self.train_jsonline)
        self.check_json_line(self.predict_jsonline, future_proj_len=self.prediction_length)

        # Writing jsonline files on S3
        # Train dataset
        path = f"s3://{self.bucket}/{self.paths['run_input']}train_{self.cutoff}.json"
        with fs.open(path, 'w') as fp:
            fp.write(self.train_jsonline)
        print(f"Uploaded dataset to {path}")

        # Test dataset
        path = f"s3://{self.bucket}/{self.paths['run_input']}predict_{self.cutoff}.json"
        with fs.open(path, 'w') as fp:
            fp.write(self.predict_jsonline)
        print(f"Uploaded dataset to {path}")

    def check_json_line(self, jsonline, future_proj_len=0):
        df = pd.read_json(jsonline, orient='records', lines=True)

        # Test if target >= min_ts_len
        df['target_len'] = df.apply(lambda x: len(x['target']), axis=1)
        test = df['target_len'] >= self.min_ts_len
        assert test.groupby(test).size().loc[True] == df.shape[0], 'Some models have a `target` less than `min_ts_len`'

        if self.cat_cols:
            # Test if right number of categorical features
            df['cat_feat_nb'] = df.apply(lambda x: len(x['cat']), axis=1)
            test = df['cat_feat_nb'] == len(self.params['cat_cols'])
            assert test.groupby(test).size().loc[True] == df.shape[0], \
                'Some models don\'t have the right number of categorical features'

        if self.dyn_cols:
            # Test if right number of dynamic features
            df['dyn_feat_nb'] = df.apply(lambda x: len(x['dynamic_feat']), axis=1)
            test = df[['dyn_feat_nb']] == len(self.params['dyn_cols'].keys())
            assert test[test['dyn_feat_nb'] == True].shape[0] == df.shape[0], \
                'Some models don\'t have the right number of dynamic features'

            # Test if right length of dynamic features
            test_dict = {}
            for i, j in enumerate(self.params['dyn_cols'].keys()):
                df[f'{j}_len'] = df.apply(lambda x: len(x['dynamic_feat'][i]), axis=1)
                df[f'{j}_len_test'] = df[f'{j}_len'] == df['target_len'] + future_proj_len
                test_dict[j] = f'{j}_len_test'

            for i in test_dict.keys():
                assert df[df[test_dict[i]] == True].shape[0] == df.shape[0], \
                    f'Some models don\'t have the right dynamic feature length for feature {i}'

    def _generate_target_data(self):

        start_time = time.time()

        # List MRP valid models
        df_mrp_valid_model = self.df_mrp.loc[self.df_mrp['mrp'].isin([2, 5]), ['model']]

        # Create df_train
        df_train = pd.merge(self.df_actual_sales, df_mrp_valid_model)  # mrp valid filter
        df_train = self._pad_to_cutoff(df_train, self.cutoff)          # pad sales to cutoff

        # Rec histo
        df_train = self._history_reconstruction(df_train, self.hist_rec_method, self.min_ts_len)

        # Add and encode cat features
        df_train = pd.merge(df_train, self.df_model_info[['model'] + self.cat_cols])

        for c in self.cat_cols:
            le = LabelEncoder()
            df_train[c] = le.fit_transform(df_train[c])

        # Save train as attribute
        self.df_train = df_train

        # Save attribute `df_predict` from `df_train`
        self.df_predict = df_train

        delta_time = str(int(time.time() - start_time))
        print(
            f"    Target data generated in {delta_time} seconds with history reconstruction method {self.hist_rec_method}")

    def _generate_dyn_feature_data(self):

        dyn_train = {}
        dyn_predict = {}

        for feat in self.dyn_cols:

            start_time = time.time()

            dyn_feat_handler = dynamic_feature_handler(
                bucket=self.bucket,
                cutoff=self.cutoff,
                feat_name=feat,
                feat_path=self.params['dyn_cols'][feat]['path'],
                hist_rec_method=self.params['dyn_cols'][feat]['hist_rec_method'],
                proj_method=self.params['dyn_cols'][feat]['proj_method'],
                proj_length=self.prediction_length
            )
            dyn_feat_handler.import_input_datasets()
            dyn_feat_handler.history_reconstruction()
            dyn_feat_handler.future_projection()

            dyn_train[feat] = dyn_feat_handler.df_hist
            dyn_predict[feat] = dyn_feat_handler.df_future

            delta_time = str(int(time.time() - start_time))
            print(f"    Generated dynamic feature {feat} in {delta_time} seconds, " +
                  f"with history reconstruction method {self.params['dyn_cols'][feat]['hist_rec_method']} " +
                  f"and future projection method {self.params['dyn_cols'][feat]['proj_method']}")

        self.dyn_train = dyn_train
        self.dyn_predict = dyn_predict

    def _history_reconstruction(self, df, hist_rec_method, min_ts_len):

        if hist_rec_method == 'cluster_avg':
            df_rec = self._hist_rec_clust_avg(df, self.df_actual_sales, self.df_model_info, min_ts_len,
                                              self.params['cluster_keys'], self.params['patch_covid'])
            return df_rec

        elif hist_rec_method is None:
            print(f"    No history reconstruction applied.")
            return df

        else:
            print(f"    History reconstruction {hist_rec_method} not implemented at the time.")

    def _hist_rec_clust_avg(
            self, df, df_actual_sales, df_model_info, min_ts_len, cluster_keys=['family_label'],
            patch_covid=True):

        covid_week_id = np.array([202011, 202012, 202013, 202014, 202015, 202016, 202017, 202018, 202019, 202020])

        # Create a complete TS dataframe
        all_model = df['model'].sort_values().unique()
        all_week = df_actual_sales \
            .loc[df_actual_sales['week_id'] <= df['week_id'].max(), 'week_id'] \
            .sort_values() \
            .unique()

        w, m = pd.core.reshape.util.cartesian_product([all_week, all_model])

        complete_ts = pd.DataFrame({'model': m, 'week_id': w})

        # Add dates
        complete_ts['ds'] = ut.week_id_to_date(complete_ts['week_id'])

        # Add cluster_keys info from df_model_info
        complete_ts = pd.merge(complete_ts, df_model_info[['model'] + cluster_keys], how='left')
        # /!\ in very rare cases, the models are too old or too recent and do not have descriptions in d_sku
        complete_ts.dropna(subset=cluster_keys, inplace=True)

        # Add current sales from df
        complete_ts = pd.merge(complete_ts, df, how='left')

        # Calculate the average sales per cluster and week from df_actual_sales
        all_sales = pd.merge(df_actual_sales, df_model_info[['model'] + cluster_keys], how='left')
        all_sales.dropna(subset=cluster_keys, inplace=True)
        all_sales = all_sales.groupby(cluster_keys + ['week_id', 'ds']) \
            .agg(mean_cluster_y=('y', 'mean')) \
            .reset_index()

        # Ad it to complete_ts
        complete_ts = pd.merge(complete_ts, all_sales, how='left')

        # Patch covid
        if patch_covid:
            # Except for models sold only during the covid period...
            exceptions = complete_ts \
                .loc[~complete_ts['week_id'].isin(covid_week_id)] \
                .groupby('model', as_index=False)['y'].sum()
            exceptions = exceptions.loc[exceptions['y'] == 0, 'model'].unique()
            
            # ...replace mean cluster sales by the last year values...
            complete_ts.loc[(complete_ts['week_id'].isin(covid_week_id)) &
                            (~complete_ts['model'].isin(exceptions)), ['mean_cluster_y']] = \
                complete_ts.loc[(complete_ts['week_id'].isin(covid_week_id - 100)) &
                                (~complete_ts['model'].isin(exceptions)), ['mean_cluster_y']].values
            
            # ...and nullify sales during Covid
            complete_ts.loc[(complete_ts['week_id'].isin(covid_week_id)) & 
                            (~complete_ts['model'].isin(exceptions)), ['y']] = np.nan

        # Compute the scale factor by row
        complete_ts['row_scale_factor'] = complete_ts['y'] / complete_ts['mean_cluster_y']

        # Compute the scale factor by model
        model_scale_factor = complete_ts \
            .groupby('model') \
            .agg(model_scale_factor=('row_scale_factor', 'mean')) \
            .reset_index()

        complete_ts = pd.merge(complete_ts, model_scale_factor, how='left')

        # have each model a scale factor?
        assert complete_ts[complete_ts.model_scale_factor.isnull()].shape[0] == 0

        # Compute a fake Y by row (if unknow fill by 0)
        complete_ts['fake_y'] = complete_ts['mean_cluster_y'] * complete_ts['model_scale_factor']
        complete_ts['fake_y'] = complete_ts['fake_y'].fillna(0).astype(int)

        # Calculate real age & total length of each TS
        ts_start_end_date = complete_ts \
            .loc[complete_ts['y'].notnull()] \
            .groupby(['model']) \
            .agg(start_date=('ds', 'min'),
                 end_date=('ds', 'max')) \
            .reset_index()

        complete_ts = pd.merge(complete_ts, ts_start_end_date, how='left')

        complete_ts['age'] = ((pd.to_datetime(complete_ts['ds']) -
                               pd.to_datetime(complete_ts['start_date'])) /
                              np.timedelta64(1, 'W')).astype(int) + 1

        complete_ts['length'] = ((pd.to_datetime(complete_ts['end_date']) -
                                  pd.to_datetime(complete_ts['ds'])) /
                                 np.timedelta64(1, 'W')).astype(int) + 1

        # Estimate the implementation period: while fake y > y
        complete_ts['is_y_sup'] = complete_ts['y'] > complete_ts['fake_y']

        end_impl_period = complete_ts[complete_ts['is_y_sup'] == True] \
            .groupby('model') \
            .agg(end_impl_period=('age', 'min')) \
            .reset_index()

        complete_ts = pd.merge(complete_ts, end_impl_period, how='left')

        # Update y from 'min_ts_len' weeks ago to the end of the implementation period
        complete_ts['y'] = np.where(
            ((complete_ts['length'] <= min_ts_len) & (complete_ts['age'] <= 0)) |
            ((complete_ts['length'] <= min_ts_len) & (complete_ts['age'] > 0) &
             (complete_ts['age'] < complete_ts['end_impl_period'])),
            complete_ts['fake_y'],
            complete_ts['y'])

        if patch_covid:
            complete_ts['y'] = np.where(
                complete_ts['week_id'].isin(covid_week_id),
                complete_ts['fake_y'],
                complete_ts['y'])

        complete_ts = complete_ts[list(df)].dropna(subset=['y']).reset_index(drop=True)
        complete_ts['y'] = complete_ts['y'].astype(int)

        return complete_ts

    def _merge_target_and_features(self, df, feat_dict=None):

        df_json_start = df.groupby(['model']).apply(lambda x: min(
            x['ds']).strftime('%Y-%m-%d %H:%M:%S')).reset_index(name='start')
        df_json_target = df.groupby(['model'])['y'].apply(list).reset_index(name='target')
        df_json_cat_cols = df[['model'] + self.cat_cols].drop_duplicates()
        df_json_cat_cols['cat'] = df_json_cat_cols.apply(lambda x: [x[i] for i in self.cat_cols], axis=1).to_frame()
        df_json_cat_cols = df_json_cat_cols[['model', 'cat']]

        df_json = pd.merge(df_json_start, df_json_target, on='model', how='inner')
        df_json = pd.merge(df_json, df_json_cat_cols, on='model', how='inner')

        df_json = df_json[['model', 'start', 'cat', 'target']]

        if feat_dict:
            for feat in feat_dict.keys():

                if 'model' in feat_dict[feat].columns:
                    df_feat = feat_dict[feat]
                else:
                    df_feat = feat_dict[feat]
                    # duplicates feat for each model
                    df_model = df[['model']].drop_duplicates()
                    df_model.insert(0, 'key', 42)
                    df_feat.insert(0, 'key', 42)
                    df_feat = pd.merge(df_model, df_feat, on='key', how='inner')

                # limit feat history based on model history
                df_model_min_week = df.groupby('model').agg(min_week_id=('week_id', min)).reset_index()
                df_feat = pd.merge(df_feat, df_model_min_week, on='model', how='inner')
                df_feat = df_feat[df_feat['week_id'] >= df_feat['min_week_id']]

                # Merge to df json
                df_json = pd.merge(df_json,
                                   df_feat.groupby(['model'])[feat].apply(list).reset_index(name=feat),
                                   on='model',
                                   how='inner')

            df_json['dynamic_feat'] = df_json.apply(lambda x: [x[i] for i in feat_dict.keys()], axis=1)
            df_json = df_json[['model', 'start', 'cat', 'target', 'dynamic_feat']]

        # Setting model at str
        df_json.loc[:, 'model'] = df_json.loc[:, 'model'].astype(str)

        nb_models = df['model'].nunique() - df_json['model'].nunique()
        print(f"    Dataset file generated. Lost {nb_models} merging features.")

        return df_json

    def _pad_to_cutoff(self, df_ts, cutoff, col='y'):

        # Add the cutoff weekend to all models to put a limit for the bfill
        models = df_ts['model'].unique()
        test_cutoff_date = ut.week_id_to_date(cutoff)
        md, cu = pd.core.reshape.util.cartesian_product([models, [cutoff]])
        df_ts_tail = pd.DataFrame({"model": md, "week_id": cu})
        df_ts_tail['ds'] = test_cutoff_date
        df_ts_tail[col] = 0
        df = df_ts.append(df_ts_tail)

        # Backfill for the cutoff week
        df = df.set_index('ds').groupby('model').resample('1W').asfreq().fillna(0)

        # Getting the df back to its original form
        df.drop(['model'], axis=1, inplace=True)
        df.reset_index(inplace=True)
        df['week_id'] = ut.date_to_week_id(df['ds'])

        # Getting rid of the cutoff week
        df = df[df['week_id'] < cutoff]
        df[col] = df[col].astype(int)

        return df

    def _shuffle_dataset(self, df):

        # Random shuffling before writing
        groups = [ts for m, ts in df.groupby('model')]
        random.shuffle(groups)

        return pd.concat(groups).reset_index(drop=True)

    def _df_to_jsonline(self, df):

        return df.to_json(orient='records', lines=True)
