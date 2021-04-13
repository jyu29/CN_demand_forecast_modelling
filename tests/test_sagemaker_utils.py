import pandas as pd
import pytest
from pytest import mark

from src.sagemaker_utils import generate_df_jobs

RUN_NAME = 'test'
ALGORITHM = 'test_algorithm'
DF_JOBS_PATH = 'tests/data/nominal_df_jobs.csv'
REFINED_DATA_SPECIFIC_PATH = 's3://fcst-refined-demand-forecast-dev/specific/'


@mark.sagemaker_utils
class generateDfJobTests():
    def test_nominal_unique(self, mocker):
        mocker.patch(
            # api_call is from slow.py but imported to main.py
            'src.sagemaker_utils._get_timestamp',
            return_value="-2021-04-13-12-24-36-826"
        )
        list_cutoff = [201905, 202004]

        expected_df_jobs = pd.read_csv(DF_JOBS_PATH, sep=';')
        df_jobs = generate_df_jobs(list_cutoff=list_cutoff,
                                   run_name=RUN_NAME,
                                   algorithm=ALGORITHM,
                                   refined_data_specific_path=REFINED_DATA_SPECIFIC_PATH
                                   )

        try:
            assert df_jobs.equals(expected_df_jobs)
        except AssertionError:
            pytest.fail("Test failed on nominal case")

    def test_format_str(self):
        list_cutoff = '202103'

        with pytest.raises(AssertionError):
            generate_df_jobs(list_cutoff=list_cutoff,
                             run_name=RUN_NAME,
                             algorithm=ALGORITHM,
                             refined_data_specific_path=REFINED_DATA_SPECIFIC_PATH
                             )

    def test_wrong_week_format(self):
        list_cutoff = [201905, '201906']

        with pytest.raises(AssertionError):
            generate_df_jobs(list_cutoff=list_cutoff,
                             run_name=RUN_NAME,
                             algorithm=ALGORITHM,
                             refined_data_specific_path=REFINED_DATA_SPECIFIC_PATH
                             )

    def test_wrong_week_iso(self):
        list_cutoff = [201905, 202065]

        with pytest.raises(AssertionError):
            generate_df_jobs(list_cutoff=list_cutoff,
                             run_name=RUN_NAME,
                             algorithm=ALGORITHM,
                             refined_data_specific_path=REFINED_DATA_SPECIFIC_PATH
                             )

    def test_wrong_run_name_regex(self):
        list_cutoff = [201905]
        run_name = "test_run"

        with pytest.raises(AssertionError):
            generate_df_jobs(list_cutoff=list_cutoff,
                             run_name=run_name,
                             algorithm=ALGORITHM,
                             refined_data_specific_path=REFINED_DATA_SPECIFIC_PATH
                             )



