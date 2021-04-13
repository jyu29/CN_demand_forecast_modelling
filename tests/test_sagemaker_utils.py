import pandas as pd
import pytest
from pytest import mark

from src.sagemaker_utils import generate_df_jobs

RUN_NAME = 'test'
ALGORITHM = 'test_algorithm'
DF_JOBS_PATH = 'tests/data/nominal_df_jobs.csv'


@mark.sagemaker_utils
class generateDfJobTests():
    def test_nominal_unique(self, mocker):
        mocker.patch(
            # api_call is from slow.py but imported to main.py
            'src.sagemaker_utils._get_timestamp',
            return_value="-2021-04-13-12-24-36-826"
        )
        list_cutoff = [201905, 202004]
        refined_data_specific_path = 's3://fcst-refined-demand-forecast-dev/specific/'

        expected_df_jobs = pd.read_csv(DF_JOBS_PATH, sep=';')
        df_jobs = generate_df_jobs(list_cutoff=list_cutoff,
                                   run_name=RUN_NAME,
                                   algorithm=ALGORITHM,
                                   refined_data_specific_path=refined_data_specific_path
                                   )

        try:
            assert df_jobs.equals(expected_df_jobs)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    # def test_format_str(self):
    #     week_id = '202103'

    #     with pytest.raises(AssertionError):
    #         week_id_to_date(week_id)
