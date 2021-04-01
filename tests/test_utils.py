from datetime import date

import numpy as np
import pandas as pd
import pytest
from pytest import mark

from src.utils import week_id_to_date, is_iso_format


@mark.utils
class weekIdToDateTests():
    def test_nominal_unique(self):
        week_id = 202103
        expected_date = date(2021, 1, 17)

        try:
            assert week_id_to_date(week_id) == expected_date
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_npint(self):
        week_id = np.int(202103)
        expected_date = date(2021, 1, 17)

        try:
            assert week_id_to_date(week_id) == expected_date
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_pdseries(self):
        week_ids = pd.Series([202103, 202250, 205824])
        expected_dates = [date(2021, 1, 17),
                          date(2022, 12, 11),
                          date(2058, 6, 9)
                          ]

        try:
            assert all(week_id_to_date(week_ids) == expected_dates)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_list(self):
        week_ids = [202103, 202250, 205824]
        expected_dates = [date(2021, 1, 17),
                          date(2022, 12, 11),
                          date(2058, 6, 9)
                          ]

        try:
            assert all(week_id_to_date(week_ids) == expected_dates)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_special_weeks(self):
        week_ids = [202052, 202053, 202101, 201952, 202001, 202152, 202201, 202252, 202301]
        expected_dates = [date(2020, 12, 20),
                          date(2020, 12, 27),
                          date(2021, 1, 3),
                          date(2019, 12, 22),
                          date(2019, 12, 29),
                          date(2021, 12, 26),
                          date(2022, 1, 2),
                          date(2022, 12, 25),
                          date(2023, 1, 1)
                          ]
        try:
            assert all(week_id_to_date(week_ids) == expected_dates)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_format_str(self):
        week_id = '202103'

        with pytest.raises(AssertionError):
            week_id_to_date(week_id)

    def test_wrong_week(self):
        week_ids = [202100, 20143, 199901]

        for w in week_ids:
            with pytest.raises(AssertionError):
                week_id_to_date(w)


@mark.utils
class isIsoFormatTests():
    def test_nominal(self):
        week_ids = [202103, 202250, 205824]

        for w in week_ids:
            try:
                assert is_iso_format(w)
            except AssertionError:
                pytest.fail("Test failed on nominal case")

    def test_wrong_weeks(self):
        week_ids = [202100, 20143, 199901]

        for w in week_ids:
            try:
                assert not is_iso_format(w)
            except AssertionError:
                pytest.fail("Test failed on nominal case")

    def test_wrong_type(self):
        week_id = '202001'

        with pytest.raises(AssertionError):
            is_iso_format(week_id)
