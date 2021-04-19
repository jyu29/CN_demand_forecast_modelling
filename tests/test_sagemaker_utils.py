import pandas as pd
import re
import os
import pytest
from unittest.mock import patch
from pytest import mark
from shutil import copyfile
import src

from src.sagemaker_utils import generate_df_jobs, _get_timestamp, import_sagemaker_params, SagemakerHandler

LIST_CUTOFF = [201905, 202004]
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

        expected_df_jobs = pd.read_csv(DF_JOBS_PATH, sep=';')
        df_jobs = generate_df_jobs(list_cutoff=LIST_CUTOFF,
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
        run_name = "test_run"

        with pytest.raises(AssertionError):
            generate_df_jobs(list_cutoff=LIST_CUTOFF,
                             run_name=run_name,
                             algorithm=ALGORITHM,
                             refined_data_specific_path=REFINED_DATA_SPECIFIC_PATH
                             )


@mark.sagemaker_utils
class GetTimestampTests:
    def test_nominal(self):
        REGEX = "^-20[1-9][1-9]-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])-([0-1][0-9]|2[0-3])-([0-5][0-9]-){2}[0-9]{3}$"  # noqa: E501
        rule = re.compile(REGEX)
        timestamp = _get_timestamp()

        try:
            assert bool(rule.match(timestamp))
        except AssertionError:
            pytest.fail("Test failed on nominal case")


@mark.sagemaker_utils
class ImportSagemakerParamsTests:
    def test_nominal(self):
        assert os.path.isfile(os.path.join('tests', 'data', 'test_config.yml')), \
            "Test configuration file missing in tests/data/"
        copyfile(os.path.join('tests', 'data', 'test_config.yml'), os.path.join('config', 'test_config.yml'))

        params = import_sagemaker_params('test_config')
        try:
            assert isinstance(params, (dict))
        except AssertionError:
            pytest.fail("Test failed on nominal case.")

        os.remove(os.path.join('config', 'test_config.yml'))

    def test_missing_env(self):
        env = 'missing_env'
        assert not os.path.isfile(os.path.join('config', f'{env}.yml'))

        with pytest.raises(AssertionError):
            import_sagemaker_params(env)


class SagemakerHandlerTests:
    @patch('src.sagemaker_utils.sagemaker.estimator.Estimator')
    @patch.object(src.sagemaker_utils.sagemaker.estimator.Estimator, 'fit')
    @patch('src.sagemaker_utils.sagemaker.Session')
    def test_nominal(self,
                     session_mocker,
                     fit_estimator_mocker,
                     estimator_mocker
                     ):

        d = {'TrainingJobStatus': 'Completed'}
        session_mocker.return_value.describe_training_job.return_value.__getitem__.side_effect = d.__getitem__
        estimator_mocker.return_value.latest_training_job.job_name = 'foo'

        params = {'run_name': 'test-sm'}
        params['df_jobs'] = generate_df_jobs(list_cutoff=LIST_CUTOFF,
                                             run_name=RUN_NAME,
                                             algorithm=ALGORITHM,
                                             refined_data_specific_path=REFINED_DATA_SPECIFIC_PATH
                                             )
        params.update(import_sagemaker_params('testing'))

        sh = SagemakerHandler(**params)

        try:
            sh.launch_training_jobs()
        except Exception:
            pytest.fail("Test failed on nominal case.")
