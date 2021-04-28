from datetime import date

import json
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from src.utils import (week_id_to_date, is_iso_format, check_list_cutoff, check_run_name,
                       check_environment)


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


class CheckCutoffListTests:
    def test_nominal_onecutoff(self):
        list_cutoff = [202001]

        try:
            assert check_list_cutoff(list_cutoff) == list_cutoff
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_twocutoff(self):
        list_cutoff = [202001, 202003]

        try:
            assert check_list_cutoff(list_cutoff) == list_cutoff
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_today(self, mocker):
        mocker.patch(
            # api_call is from slow.py but imported to main.py
            'src.utils.get_current_week',
            return_value=202117
        )
        list_cutoff = 'today'

        try:
            assert check_list_cutoff(list_cutoff) == [202117]
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_str_onecutoff(self):
        list_cutoff = '[202110]'

        try:
            assert check_list_cutoff(list_cutoff) == [202110]
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_str_twocutoff(self):
        list_cutoff = '[202110, 201940]'

        try:
            assert check_list_cutoff(list_cutoff) == [202110, 201940]
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_str_nolist(self):
        list_cutoff = '201805'

        try:
            assert check_list_cutoff(list_cutoff) == [201805]
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_mispelled_today(self):
        list_cutoff = 'toaday'

        with pytest.raises(json.decoder.JSONDecodeError):
            check_list_cutoff(list_cutoff)

    def test_wrong_iso_weeks(self):
        list_cutoff = [202067, 29134]

        with pytest.raises(AssertionError):
            check_list_cutoff(list_cutoff)

    def test_wrong_iso_weeks_str(self):
        list_cutoff = '[202067, 29134]'

        with pytest.raises(AssertionError):
            check_list_cutoff(list_cutoff)


class CheckRunNameTests:
    def test_nominal_case(self):
        run_name = 'testname'

        try:
            check_run_name(run_name)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_case_with_characters(self):
        run_name = 'test-name'

        try:
            check_run_name(run_name)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_nominal_case_with_number(self):
        run_name = 'test-name2'

        try:
            check_run_name(run_name)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_underscore(self):
        run_name = 'test_name'

        with pytest.raises(AssertionError):
            check_run_name(run_name)

    def test_too_long(self):
        run_name = 'testnamsdofuhsdlfjshdflkusdgflsdkfjbqwleifgslkausdgfasfasgdfdskjhgdwfe'

        with pytest.raises(AssertionError):
            check_run_name(run_name)


class CheckEnvironmentTests:
    def test_nominal(self):
        environment = 'testing'

        try:
            environment = check_environment(environment)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_no_config(self):
        environment = 'doesntexist'

        with pytest.raises(AssertionError):
            check_environment(environment)

    def test_wrong_type(self):
        environment = 1234

        with pytest.raises(AssertionError):
            check_environment(environment)
