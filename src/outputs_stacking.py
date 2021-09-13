import logging

import pandas as pd
import numpy as np

from src.utils import (from_uri, read_jsonline_s3, read_multipart_parquet_s3, write_df_to_parquet_on_s3)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def read_format_stacking_data(df_jobs_cutoff):

    assert df_jobs_cutoff['cutoff'].nunique() == 1, '`df_jobs_cutoff` must contain only one cutoff'

    deepar_input = df_jobs_cutoff[df_jobs_cutoff['algorithm'] == 'deepar']['predict_path'].values[0]
    deepar_output = deepar_input.replace('input', 'output') + '.out'

    hw_input = df_jobs_cutoff.loc[df_jobs_cutoff['algorithm'] == 'hw']['predict_path'].values[0]
    hw_output = hw_input.replace('input', 'output') + '.out'

    deepar_hw_output = hw_output.replace('hw', 'deepar-hw')

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

    # Load & format hw forecast
    logger.debug("Load & format hw forecasts...")

    hw = read_multipart_parquet_s3(*from_uri(hw_output))
    hw.rename(columns={'forecast': 'forecast_hw'}, inplace=True)

    return deepar, hw, deepar_hw_output


def compute_stacking(deepar, hw, stacking_start, stacking_stop, nb_stacking_weeks):
    # Calculate deepar_hw forecast with a smooth stacking
    deepar_hw = hw.merge(deepar, how='left')

    deepar_hw['smooth_weight'] = (deepar_hw['forecast_step'] - stacking_start) / nb_stacking_weeks

    deepar_hw['forecast'] = np.where(
        deepar_hw['forecast_step'] <= stacking_start,
        deepar_hw['forecast_deepar'],
        np.where(
            deepar_hw['forecast_step'] <= stacking_stop,
            deepar_hw['forecast_deepar'] * (1 - deepar_hw['smooth_weight']) + \
            deepar_hw['forecast_hw'] * deepar_hw['smooth_weight'],
            deepar_hw['forecast_hw']
        )
    ).round().astype(int)

    # Format deepar_hw
    deepar_hw = deepar_hw[['model_id', 'forecast_step', 'forecast']]

    return deepar_hw


def calculate_outputs_stacking(df_jobs, smooth_stacking_range=(10, 16)):
    """
    """
    assert all(a in df_jobs['algorithm'].unique() for a in ['deepar', 'hw']), \
        "Deepar & hw must be included in the algorithms launched in the previous steps."

    assert (all(isinstance(v, int) for v in smooth_stacking_range)) & \
           (smooth_stacking_range[0] > 1) & \
           (smooth_stacking_range[1] > smooth_stacking_range[0]), \
           "The stacking range should be a tuple of 2 integer respecting (a, b): 1 < a < b."

    list_cutoff = df_jobs['cutoff'].unique()
    stacking_start = smooth_stacking_range[0]
    stacking_stop = smooth_stacking_range[1]
    nb_stacking_weeks = stacking_stop - stacking_start

    for cutoff in list_cutoff:

        logger.info(f"Calculate deepar_hw for cutoff {cutoff}...")

        # Set needed path
        df_jobs_cutoff = df_jobs[df_jobs['cutoff'] == cutoff].copy()

        deepar, hw, deepar_hw_output = read_format_stacking_data(df_jobs_cutoff)

        assert deepar['forecast_step'].max() >= stacking_stop, \
            f"Deepar's forecast horizon is too short to apply stacking. Minimum horizon required: {stacking_stop}"

        assert hw['forecast_step'].max() >= stacking_stop, \
            f"hw's forecast horizon is too short to apply stacking. Minimum horizon required: {stacking_stop}"

        deepar_hw = compute_stacking(deepar, hw, stacking_start, stacking_stop, nb_stacking_weeks)

        write_df_to_parquet_on_s3(deepar_hw, *from_uri(deepar_hw_output), verbose=True)
