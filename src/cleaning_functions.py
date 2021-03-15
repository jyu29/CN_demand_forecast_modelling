import datetime as dt

import numpy as np
import pandas as pd

from src.utils import date_to_week_id, week_id_to_date


def pad_to_cutoff(df_ts, cutoff, col='sales_quantity'):

    # Add the cutoff weekend to all models to put a limit for the bfill
    models = df_ts['model_id'].unique()
    test_cutoff_date = week_id_to_date(cutoff)
    md, cu = pd.core.reshape.util.cartesian_product([models, [cutoff]])
    df_ts_tail = pd.DataFrame({"model_id": md, "week_id": cu})
    df_ts_tail['date'] = test_cutoff_date
    df_ts_tail[col] = 0
    df = df_ts.append(df_ts_tail)

    # Backfill for the cutoff week
    df = df.set_index('date').groupby('model_id').resample('1W').asfreq().fillna(0)

    # Getting the df back to its original form
    df.drop(['model_id'], axis=1, inplace=True)
    df.reset_index(inplace=True)
    df['week_id'] = date_to_week_id(df['date'])

    # Getting rid of the cutoff week
    df = df[df['week_id'] < cutoff]
    df[col] = df[col].astype(int)

    return df


def cold_start_rec(df,
                   df_model_week_sales,
                   df_model_week_tree,
                   rec_cold_start_length,
                   patch_covid_weeks,
                   rec_cold_start_group=['family_label'],
                   patch_covid=True
                   ):

    # Create a complete TS dataframe
    all_model = df['model_id'].sort_values().unique()
    all_week = df_model_week_sales \
        .loc[df_model_week_sales['week_id'] <= df['week_id'].max(), 'week_id'] \
        .sort_values() \
        .unique()

    w, m = pd.core.reshape.util.cartesian_product([all_week, all_model])

    complete_ts = pd.DataFrame({'model_id': m, 'week_id': w})

    # Add dates
    complete_ts['date'] = week_id_to_date(complete_ts['week_id'])

    # Add cluster_keys info from df_model_week_tree
    complete_ts = pd.merge(complete_ts, df_model_week_tree[['model_id'] + rec_cold_start_group], how='left')
    # /!\ in very rare cases, the models are too old or too recent and do not have descriptions in d_sku
    complete_ts.dropna(subset=rec_cold_start_group, inplace=True)

    # Add current sales from df
    complete_ts = pd.merge(complete_ts, df, how='left')

    # Calculate the average sales per cluster and week from df_model_week_sales
    all_sales = pd.merge(df_model_week_sales, df_model_week_tree[['model_id'] + rec_cold_start_group], how='left')
    all_sales.dropna(subset=rec_cold_start_group, inplace=True)
    all_sales = all_sales.groupby(rec_cold_start_group + ['week_id', 'date']) \
        .agg(mean_cluster_sales_quantity=('sales_quantity', 'mean')) \
        .reset_index()

    # Ad it to complete_ts
    complete_ts = pd.merge(complete_ts, all_sales, how='left')

    # Patch covid
    if patch_covid:
        print(f"Covid-19 history reconstruction patch applied on weeks {patch_covid_weeks}")
        
        # Identify the Covid weeks present in complete_ts
        covid_week_id = np.intersect1d(complete_ts['week_id'].unique(),
                                       np.array(patch_covid_weeks))
        
        # Except for models sold only during the covid period...
        exceptions = complete_ts \
            .loc[~complete_ts['week_id'].isin(covid_week_id)] \
            .groupby('model_id', as_index=False)['sales_quantity'].sum()
        exceptions = exceptions.loc[exceptions['sales_quantity'] == 0, 'model_id'].unique()
        
        # ...replace mean cluster sales by the last year values...
        complete_ts.loc[(complete_ts['week_id'].isin(covid_week_id)) &
                        (~complete_ts['model_id'].isin(exceptions)), ['mean_cluster_sales_quantity']] = \
            complete_ts.loc[(complete_ts['week_id'].isin(covid_week_id - 100)) &
                            (~complete_ts['model_id'].isin(exceptions)), ['mean_cluster_sales_quantity']].values
        
        # ...and nullify sales during Covid
        complete_ts.loc[(complete_ts['week_id'].isin(covid_week_id)) & 
                        (~complete_ts['model_id'].isin(exceptions)), ['sales_quantity']] = np.nan

    # Compute the scale factor by row
    complete_ts['row_scale_factor'] = complete_ts['sales_quantity'] / complete_ts['mean_cluster_sales_quantity']

    # Compute the scale factor by model
    model_scale_factor = complete_ts \
        .groupby('model_id') \
        .agg(model_scale_factor=('row_scale_factor', 'mean')) \
        .reset_index()

    complete_ts = pd.merge(complete_ts, model_scale_factor, how='left')

    # have each model a scale factor?
    assert complete_ts[complete_ts.model_scale_factor.isnull()].shape[0] == 0

    # Compute a fake sales quantity by row (if unknow fill by 0)
    complete_ts['fake_sales_quantity'] = complete_ts['mean_cluster_sales_quantity'] * complete_ts['model_scale_factor']
    complete_ts['fake_sales_quantity'] = complete_ts['fake_sales_quantity'].fillna(0).astype(int)

    # Calculate real age & total length of each TS
    ts_start_end_date = complete_ts \
        .loc[complete_ts['sales_quantity'].notnull()] \
        .groupby(['model_id']) \
        .agg(start_date=('date', 'min'),
             end_date=('date', 'max')) \
        .reset_index()

    complete_ts = pd.merge(complete_ts, ts_start_end_date, how='left')

    complete_ts['age'] = ((pd.to_datetime(complete_ts['date']) -
                           pd.to_datetime(complete_ts['start_date'])) /
                          np.timedelta64(1, 'W')).astype(int) + 1

    complete_ts['length'] = ((pd.to_datetime(complete_ts['end_date']) -
                              pd.to_datetime(complete_ts['date'])) /
                             np.timedelta64(1, 'W')).astype(int) + 1

    # Estimate the implementation period: while fake sales quantity > sales quantity
    complete_ts['is_sales_quantity_sup'] = complete_ts['sales_quantity'] > complete_ts['fake_sales_quantity']

    end_impl_period = complete_ts[complete_ts['is_sales_quantity_sup']] \
        .groupby('model_id') \
        .agg(end_impl_period=('age', 'min')) \
        .reset_index()

    complete_ts = pd.merge(complete_ts, end_impl_period, how='left')

    # Update sales quantity from 'rec_cold_start_length' weeks ago to the end of the implementation period
    cond = ((complete_ts['length'] <= rec_cold_start_length) & (complete_ts['age'] <= 0)) | \
           ((complete_ts['length'] <= rec_cold_start_length) & (complete_ts['age'] > 0) &
            (complete_ts['age'] < complete_ts['end_impl_period']))
    complete_ts['sales_quantity'] = np.where(cond, complete_ts['fake_sales_quantity'], complete_ts['sales_quantity'])
    complete_ts['is_rec'] = np.where(cond, 1, 0)

    if patch_covid:
        cond = complete_ts['week_id'].isin(covid_week_id)
        complete_ts['sales_quantity'] = np.where(cond, complete_ts['fake_sales_quantity'], complete_ts['sales_quantity'])
        complete_ts['is_rec'] = np.where(cond, 1, complete_ts['is_rec'])

    # Format
    complete_ts = complete_ts[list(df) + ['is_rec']].dropna(subset=['sales_quantity']).reset_index(drop=True)
    complete_ts['sales_quantity'] = complete_ts['sales_quantity'].astype(int)

    return complete_ts


def check_weeks_df(df, min_week, cutoff, future_weeks=0, week_column='week_id'):
    """
    Checks the presence of all past & future weeks in a global feature pandas DataFrame

    Args:
        df (pandas.DataFrame): Pandas DataFrame to check
        min_week (int): Minimum week to check in the DataFrame (YYYYWW ISO Format)
        cutoff (int): first week of forecast (YYYYWW ISO Format)
        future_weeks (int): Expected number of weeks in the future
        week_column (str): Name of the column on which to make the checks
    """
    feature_name = ', '.join((set(df.columns) - set(['model_id', 'week_id'])))
    expected_week_range = generate_expected_week_range(min_week, cutoff, future_weeks)
    actual_week_range = df[week_column].unique()

    for w in expected_week_range:
        assert w in actual_week_range, f"Week {w} is not in {feature_name} dataframe."


def generate_expected_week_range(min_week, cutoff, future_weeks):
    min_date = week_id_to_date(min_week)
    cutoff_date = week_id_to_date(cutoff)
    max_date = cutoff_date + dt.timedelta(weeks=future_weeks - 1)

    date_range = pd.date_range(start=min_date, end=max_date, freq='W')
    expected_week_range = [date_to_week_id(d) for d in date_range]

    return expected_week_range


def generate_empty_dyn_feat_global(df_sales, min_week, cutoff, future_projection):
    expected_week_range = generate_expected_week_range(min_week, cutoff, future_projection)
    expected_models = df_sales['model_id'].unique()

    m, w = pd.core.reshape.util.cartesian_product([expected_models, expected_week_range])

    df_empty_dyn_feat_global = pd.DataFrame({'week_id': w,
                                             'model_id': m})

    return df_empty_dyn_feat_global
