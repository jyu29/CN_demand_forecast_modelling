import numpy as np
import pandas as pd

from src.utils import (date_to_week_id, week_id_to_date, is_iso_format)


def pad_to_cutoff(df_ts: pd.DataFrame,
                  cutoff: int,
                  col: str = 'sales_quantity'
                  ) -> pd.DataFrame:
    """
    Forward fills with zeros time series in Pandas DataFrame up to week cutoff.

    The function will complete the dataframe for all models (`model_id`) with a frequency
    of 1 week, up to week_id cutoff (excluding it) on the column `col` with value = 0.
    The input dataframe must include columns ['model_id', 'week_id', 'date'].

    Args:
        df_ts (pd.DataFrame): Timeseries DataFrame
        cutoff (int): ISO 8601 Format Week id (YYYYWW) to forward fill to
        col (str): Name of the column to fill

    Returns:
        df (pd.DataFrame): Padded DataFrame
    """
    assert is_iso_format(cutoff)
    assert isinstance(cutoff, (int, np.int64))
    assert isinstance(df_ts, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(df_ts['date'])
    assert set(df_ts.columns) == set(['model_id', 'week_id', 'date', col])

    # Limiting dataset to avoid errors if we have data further than cutoff
    df_ts = df_ts[df_ts['week_id'] < cutoff]
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
                   rec_length,
                   rec_cold_start_group=['family_label']
                   ):
    """
    Create a fake sales history for models that started too recently.

    Args:
        df (pd.DataFrame): The sales of the time series pool on which we want to apply a cold-start 
            reconstruction
        df_model_week_sales (pd.DataFrame): Sales of all available time series
        df_model_week_tree (pd.DataFrame): Tree structure information of all available time series
        rec_length (int): The minimum expected size of each time series after reconstruction 
        rec_cold_start_group (list): The list of aggregation keys (of the tree structure) on which 
            to calculate the reconstruction properties

    Returns:
        complete_ts (pd.DataFrame): The sales of the time series pool including the reconstruction
    """
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

    complete_ts['age'] = ((pd.to_datetime(complete_ts['date']) - pd.to_datetime(complete_ts['start_date'])
                           ) / np.timedelta64(1, 'W')).astype(int) + 1

    complete_ts['length'] = ((pd.to_datetime(complete_ts['end_date']) - pd.to_datetime(complete_ts['date'])
                              ) / np.timedelta64(1, 'W')).astype(int) + 1

    # Estimate the implementation period: while fake sales quantity > sales quantity
    complete_ts['is_sales_quantity_sup'] = complete_ts['sales_quantity'] > complete_ts['fake_sales_quantity']

    end_impl_period = complete_ts[complete_ts['is_sales_quantity_sup']] \
        .groupby('model_id') \
        .agg(end_impl_period=('age', 'min')) \
        .reset_index()

    complete_ts = pd.merge(complete_ts, end_impl_period, how='left')

    # Update sales quantity from 'rec_length' weeks ago to the end of the implementation period
    cond = ((complete_ts['length'] <= rec_length) & (complete_ts['age'] <= 0)
            ) | ((complete_ts['length'] <= rec_length
                  ) & (complete_ts['age'] > 0
                       ) & (complete_ts['age'] < complete_ts['end_impl_period'])
                 )
    complete_ts['sales_quantity'] = np.where(cond, complete_ts['fake_sales_quantity'], complete_ts['sales_quantity'])
    complete_ts['is_rec'] = np.where(cond, 1, 0)

    # Format
    complete_ts = complete_ts[list(df) + ['is_rec']].dropna(subset=['sales_quantity']).reset_index(drop=True)
    complete_ts['sales_quantity'] = complete_ts['sales_quantity'].astype(int)

    return complete_ts


def initialize_df_dynamic_features(df, cutoff, prediction_length):
    """
    Initialize the dynamic features dataframe on which all dynamic features will be joined.
    
    Dynamic features must be available in the future, therefore the expected size of the dataframe 
    must correspond to the observed sales for each model plus its projection over the forecast horizon.

    Args:
        df (pd.DataFrame): The sales dataframe
        cutoff (int): The week_id of the cutoff, which corresponds to the first forecast step 
        prediction_length (int): The forecast horizon

    Returns:
        df_dynamic_features (pd.DataFrame): The initialized dynamic features dataframe
    """
    assert df['week_id'].max() < cutoff, \
    "The sales dataframe must be cut before the cutoff for this function to work properly"
    
    model_ids = df['model_id'].unique()
    
    future_date_range = pd.date_range(start=week_id_to_date(cutoff), periods=prediction_length, freq='W')
    future_date_range_weeks = [date_to_week_id(w) for w in future_date_range]

    w, m = pd.core.reshape.util.cartesian_product([future_date_range_weeks, model_ids])

    future_model_week = pd.DataFrame({'model_id': m, 'week_id': w})

    df_dynamic_features = df[['model_id', 'week_id']].append(future_model_week)
    df_dynamic_features.sort_values(['model_id', 'week_id'], inplace=True)
        
    return df_dynamic_features


def is_rec_feature_processing(df, cutoff, prediction_length):
    """
    Process the dynamic features `is_rec` based on `cold_start_rec` output.
    
    Dynamic features must be available in the future, therefore the feature 
    is projected on the horizon with a value of 0.
    
    Args:
        df (pd.DataFrame): The sales dataframe which includes the reconstructed weeks information
        cutoff (int): The week_id of the cutoff, which corresponds to the first forecast step 
        prediction_length (int): The forecast horizon

    Returns:
        df_is_rec (pd.DataFrame): The 'is_rec' dynamic feature dataframe
    """
    model_ids = df['model_id'].unique()
    
    future_date_range = pd.date_range(start=week_id_to_date(cutoff), periods=prediction_length, freq='W')
    future_date_range_weeks = [date_to_week_id(w) for w in future_date_range]

    w, m = pd.core.reshape.util.cartesian_product([future_date_range_weeks, model_ids])

    df_is_rec_future = pd.DataFrame({'model_id': m, 'week_id': w})
    df_is_rec_future['is_rec'] = 0
    
    df_is_rec = df[['model_id', 'week_id', 'is_rec']].append(df_is_rec_future)

    return df_is_rec


def features_forward_fill(df, cutoff, projection_length):
    """Fills a feature dataframe in the future

    Takes a pd.DataFrame `df`, cuts it at `cutoff` (to avoid future leakage) and forward fills (ffill)
    the last feature value up to week cutoff + projection_length

    Args:
        df (pd.DataFrame): Feature dataframe (must include column 'week_id' and only one feature column)
        cutoff (int): Cutoff week in ISO 8601 Format (YYYYWW)
        projection_length (int): Number of weeks in the future on which to predict

    Returns:
        pd.DataFrame with the same structure as `df`, filled with missing week_ids & feature, with last
        value provided filled in the future
    """
    cutoff_date = week_id_to_date(cutoff)
    last_week_date = cutoff_date + pd.Timedelta(value=projection_length - 1, unit='W')
    last_week = date_to_week_id(last_week_date)

    df = df[df['week_id'] < cutoff]
    df = df.append({'week_id': last_week}, ignore_index=True).astype({'week_id': int})
    df['week_id'] = week_id_to_date(df['week_id'])
    df = df.set_index('week_id').asfreq('W')
    df = df.fillna(method='ffill')
    df.reset_index(inplace=True)
    df['week_id'] = date_to_week_id(df['week_id'])

    return df


def apply_first_lockdown_patch(df_sales, df_sales_imputed):
    """
    Replaces the lockdown weeks of `df_sales` with the imputed ones of `df_sales_imputed`.
    
    Args:
        df_sales (pd.DataFrame): The initial sales dataframe
        df_sales_imputed (pd.DataFrame): The imputed sales dataframe
        
    Returns:
        pd.DataFrame with the same structure as `df_sales`, whose lockdown weeks have been 
        replaced by the ones in `df_sales_imputed`
    """
    df_sales = pd.merge(df_sales, df_sales_imputed, how='left')
    df_sales['sales_quantity'] = np.where(df_sales['sales_quantity_imputed'].notnull(),
                                          df_sales['sales_quantity_imputed'],
                                          df_sales['sales_quantity'])
    df_sales.drop(columns='sales_quantity_imputed', inplace=True)
    return df_sales