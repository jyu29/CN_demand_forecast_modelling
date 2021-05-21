import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.model_stacking import read_format_stacking_data, compute_stacking


DATA_PATH = os.path.join('tests', 'data')
DF_JOBS_PATH = os.path.join(DATA_PATH, 'df_jobs_stacking.parquet')
# ARIMA_PREDICT_PATH = os.path.join(DATA_PATH, 'stacking-predict-arima.parquet')
# DEEPAR_INPUT_PREDICT_PATH = os.path.join(DATA_PATH, 'deepar.json')
# DEEPAR_PREDICT_PATH = os.path.join(DATA_PATH, 'stacking-predict-deepar.json')
EXPECTED_DEEPAR_STACKING = os.path.join(DATA_PATH, 'stacking_deepar.parquet')
EXPECTED_ARIMA_STACKING = os.path.join(DATA_PATH, 'stacking_arima.parquet')
EXPECTED_DEEPAR_ARIMA_STACKING = os.path.join(DATA_PATH, 'stacking_deepararima.parquet')


def check_dataframe_equality(df1, df2):
    """Checks dataframes equality
    Checks if two dataframes `df1` & `df2` on the following criterias :
    - columns name (ignoring order)
    - number of lines
    - line values (ignoring order, and tentatively trying to reconcile data types)
    """

    # Ensuring that DTypes are consistent between the two dataframes
    try:
        df2 = df2.astype(df1.dtypes)
    except TypeError:
        try:
            df1 = df1.astype(df2.dtypes)
        except AssertionError as e:
            e.args += ('Dataframes formats are incompatible, comparison is not possible',)
            raise

    # Columns order handling
    df1 = df1[list(set(df1.columns))]
    df2 = df2[list(set(df2.columns))]

    # Overall order
    df1 = df1.transform(np.sort)
    df2 = df2.transform(np.sort)

    # Resetting index
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    return df1.equals(df2)


def identity(foo):
    return foo


def read_jsonline_mocked(*jsonline_path):
    jsonline_path = ''.join(list(jsonline_path))
    df_jsonline = pd.read_json(jsonline_path, orient='records', lines=True)

    return df_jsonline


def read_parquet_mocked(*parquet_path):
    parquet_path = ''.join(list(parquet_path))
    df_jsonline = pd.read_parquet(parquet_path)

    return df_jsonline


class ReadFormatStackingDataTests:
    @patch('src.model_stacking.read_multipart_parquet_s3', read_parquet_mocked)
    @patch('src.model_stacking.read_jsonline_s3', read_jsonline_mocked)
    @patch('src.model_stacking.from_uri', identity)
    def test_nominal(self
                     # from_uri_mocker,
                     # read_json_mocker,
                     # read_parquet_mocker
                     ):

        cutoff = 202050
        df_jobs = pd.read_parquet(DF_JOBS_PATH)
        df_jobs_cutoff = df_jobs[df_jobs['cutoff'] == cutoff].copy()
        deepar, arima, _ = read_format_stacking_data(df_jobs_cutoff)

        expected_deepar = pd.read_parquet(EXPECTED_DEEPAR_STACKING)
        expected_arima = pd.read_parquet(EXPECTED_ARIMA_STACKING)

        try:
            assert check_dataframe_equality(deepar, expected_deepar)
            assert check_dataframe_equality(arima, expected_arima)
        except AssertionError:
            pytest.fail("Test failed on nominal case")


class ComputeDeeparArimaStackingTests:
    def test_nominal(self):

        deepar = pd.read_parquet(EXPECTED_DEEPAR_STACKING)
        arima = pd.read_parquet(EXPECTED_ARIMA_STACKING)
        stacking_start = 8
        stacking_stop = 16
        nb_stacking_weeks = 8

        deepar_arima = compute_stacking(deepar, arima, stacking_start, stacking_stop, nb_stacking_weeks)

        expected_deepar_arima = pd.read_parquet(EXPECTED_DEEPAR_ARIMA_STACKING)

        try:
            assert check_dataframe_equality(deepar_arima, expected_deepar_arima)
        except AssertionError:
            pytest.fail("Test failed on nominal case")
