import pandas as pd
import pytest
from pytest import mark

from src.refining_specific_functions import pad_to_cutoff


PAD_TO_CUTOFF_DATAPATH = 'tests/data/pad_to_cutoff_nominal.csv'
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
