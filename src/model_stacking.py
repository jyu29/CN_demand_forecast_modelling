import logging

import pandas as pd
import numpy as np

from src.utils import (from_uri, read_jsonline_s3, read_multipart_parquet_s3, write_df_to_parquet_on_s3)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def calculate_deepar_arima_stacking(df_jobs, smooth_stacking_range=(8, 16)):
    """
    """
    assert all(a in df_jobs['algorithm'].unique() for a in ['deepar', 'arima']), \
        "Deepar & arima must be included in the algorithms launched in the previous steps."

    assert (all(isinstance(v, int) for v in smooth_stacking_range)) & \
           (smooth_stacking_range[0] > 1) & \
           (smooth_stacking_range[1] > smooth_stacking_range[0]), \
           "The stacking range should be a tuple of 2 integer respecting (a, b): 1 < a < b."

    list_cutoff = df_jobs['cutoff'].unique()
    stacking_start = smooth_stacking_range[0]
    stacking_stop = smooth_stacking_range[1]
    nb_stacking_weeks = stacking_stop - stacking_start

    for cutoff in list_cutoff:

        logger.info(f"Calculate deepar_arima for cutoff {cutoff}...")

        # Set needed path
        df_jobs_cutoff = df_jobs[df_jobs['cutoff'] == cutoff].copy()

        deepar_input = df_jobs_cutoff.loc[df_jobs['algorithm'] == 'deepar', 'predict_path'].values[0]
        deepar_output = deepar_input.replace('input', 'output') + '.out'

        arima_input = df_jobs_cutoff.loc[df_jobs['algorithm'] == 'arima', 'predict_path'].values[0]
        arima_output = arima_input.replace('input', 'output') + '.out'

        deepar_arima_output = arima_output.replace('arima', 'deepar-arima')

        # Load & format deepar forecast
        logger.debug("Load & format Deepar forecasts...")

        deepar_model_id = read_jsonline_s3(*from_uri(deepar_input))['model_id'].values
        deepar = read_jsonline_s3(*from_uri(deepar_output)).set_index(deepar_model_id)
        deepar = pd.concat([deepar, deepar['quantiles'].apply(pd.Series)], axis=1)
        deepar.drop(columns=['quantiles'], inplace=True)

        horizon = len(deepar.iloc[0]['mean'])

        deepar = deepar.reset_index().rename(columns={'index': 'model_id'})
        deepar = deepar.apply(pd.Series.explode)
        deepar = deepar.astype(float).clip(lower=0).round().astype(int)

        deepar['forecast_step'] = list(range(1, horizon + 1)) * deepar['model_id'].nunique()
        deepar.rename(columns={'0.5': 'forecast_deepar'}, inplace=True)
        deepar = deepar[['model_id', 'forecast_step', 'forecast_deepar']]

        assert deepar['forecast_step'].max() >= stacking_stop, \
            f"Deepar's forecast horizon is too short to apply stacking. Minimum horizon required: {stacking_stop}"

        # Load & format arima forecast
        logger.debug("Load & format Arima forecasts...")

        arima = read_multipart_parquet_s3(*from_uri(arima_output))
        arima.rename(columns={'forecast': 'forecast_arima'}, inplace=True)

        assert arima['forecast_step'].max() >= stacking_stop, \
            f"Arima's forecast horizon is too short to apply stacking. Minimum horizon required: {stacking_stop}"

        # Calculate deepar_arima forecast with a smooth stacking
        deepar_arima = arima.merge(deepar, how='left')

        deepar_arima['smooth_weight'] = (deepar_arima['forecast_step'] - nb_stacking_weeks) / nb_stacking_weeks

        deepar_arima['forecast'] = np.where(
            deepar_arima['forecast_step'] <= stacking_start,
            deepar_arima['forecast_deepar'],
            np.where(
                deepar_arima['forecast_step'] <= stacking_stop,
                deepar_arima['forecast_deepar'] * (1 - deepar_arima['smooth_weight']) + \
                deepar_arima['forecast_arima'] * deepar_arima['smooth_weight'],
                deepar_arima['forecast_arima']
            )
        ).round().astype(int)

        # Format deepar_arima
        deepar_arima = deepar_arima[['model_id', 'forecast_step', 'forecast']]

        write_df_to_parquet_on_s3(deepar_arima, *from_uri(deepar_arima_output), verbose=True)
