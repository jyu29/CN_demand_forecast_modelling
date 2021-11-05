import os
import pandas as pd
import pytest
from pytest import mark
from src.utils import week_id_to_date

from src.refining_specific_functions import pad_to_cutoff, apply_cold_start_reconstruction

DATA_PATH = os.path.join('tests', 'data')
PAD_TO_CUTOFF_DATAPATH = os.path.join(DATA_PATH, 'pad_to_cutoff_nominal.csv')
COLD_START_REC_DF = os.path.join(DATA_PATH, 'cold_start_rec_df.csv')
COLD_START_REC_TREE = os.path.join(DATA_PATH, 'cold_start_rec_tree.csv')
COLD_START_REC_SALES = os.path.join(DATA_PATH, 'cold_start_rec_sales.csv')
REC_LENGTH = 104
REC_COLD_START_GROUP = ['family_id']


@mark.refining_specific_functions
class PadToCutoffTests:
    def test_nominal(self):
        CUTOFF = 202010
        COL = 'sales_quantity'
        df = pd.read_csv(filepath_or_buffer=PAD_TO_CUTOFF_DATAPATH, sep=';', parse_dates=['date'])
        df_padded = pad_to_cutoff(df, CUTOFF, col=COL)

        try:
            assert df_padded[(df_padded['model_id'] == 8553374) &
                             (df_padded['week_id'] > 201952) &
                             (df_padded['week_id'] < CUTOFF)][COL].shape == (9,)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

        try:
            assert set(df_padded[(df_padded['model_id'] == 8553374) &
                                 (df_padded['week_id'] > 201952) &
                                 (df_padded['week_id'] < CUTOFF)][COL]) == set([0])
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_cutoff_before(self):
        CUTOFF = 201940
        COL = 'sales_quantity'
        df = pd.read_csv(filepath_or_buffer=PAD_TO_CUTOFF_DATAPATH, sep=';', parse_dates=['date'])
        df_padded = pad_to_cutoff(df, CUTOFF, col=COL)

        try:
            assert df_padded[df_padded['week_id'] > CUTOFF].shape[0] == 0
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_wrong_column(self):
        CUTOFF = 202010
        COL = 'quantity'
        df = pd.read_csv(filepath_or_buffer=PAD_TO_CUTOFF_DATAPATH, sep=';', parse_dates=['date'])

        with pytest.raises(AssertionError):
            pad_to_cutoff(df, CUTOFF, col=COL)


@mark.refining_specific_functions
class ZeroPaddingRecTests:
    def test_nominal(self):
        pass


@mark.refining_specific_functions
class ColdStartRecTests:
    def test_nominal(self):
        df_sales = pd.read_csv(COLD_START_REC_DF, sep=';', parse_dates=['date'])
        df_model_week_sales = pd.read_csv(COLD_START_REC_SALES, sep=';', parse_dates=['date'])
        df_model_week_tree = pd.read_csv(COLD_START_REC_TREE, sep=';')
        df_cold_start = apply_cold_start_reconstruction(df_sales,
                                             df_model_week_sales,
                                             df_model_week_tree,
                                             REC_LENGTH,
                                             REC_COLD_START_GROUP)

        # Checking if all expected reconstruction occured
        try:
            assert df_cold_start.groupby('model_id').max()['is_rec'].to_dict() == {8366364: 0, 8497787: 1, 8524262: 1}
        except AssertionError:
            pytest.fail("Test failed on nominal case : some models that should have been reconstructed were not.")

        # Checking if all models have expected reconstruction length
        df = df_cold_start.groupby(['model_id']).agg(min_week_id=('week_id', 'min'), max_week_id=('week_id', 'max'))
        df['length'] = df.apply(lambda x: len(pd.date_range(start=week_id_to_date(x['min_week_id']),
                                                            end=week_id_to_date(x['max_week_id']), freq='1W')), axis=1)
        try:
            assert all([le >= REC_LENGTH for le in df['length'].to_list()])
        except AssertionError:
            pytest.fail("Test failed on nominal case : some models don't have the expected length "
                        "after cold start reconstruction.")
