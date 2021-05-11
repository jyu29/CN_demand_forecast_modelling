import datetime
import json
import io
import re
import os
import boto3
import s3fs
import yaml

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from uritools import urisplit
from typing import Union


def date_to_week_id(date):
    """
    Turn a date to Decathlon week id
    :param date: (str, pd.Timestamp or pd.Series) the date or pandas column of dates
    :return: (int or pd.Series) the week id or pandas column of week ids
    """
    assert isinstance(date, (str, pd.Timestamp, pd.Series, datetime.date))
    if isinstance(date, (str, pd.Timestamp, datetime.date)):
        date = pd.Timestamp(date)
        if date.dayofweek == 6:  # If sunday, replace by next monday to get the correct iso week
            date = date + pd.Timedelta(1, unit='D')
        week_id = int(str(date.isocalendar()[0]) + str(date.isocalendar()[1]).zfill(2))
        return week_id
    else:
        df = pd.DataFrame({'date': pd.to_datetime(date)})
        df['dow'] = df['date'].dt.dayofweek
        df.loc[df['dow'] == 6, 'date'] = df.loc[df['dow'] == 6, 'date'] + pd.Timedelta(1, unit='D')
        df['week_id'] = df['date'].apply(lambda x: int(str(x.isocalendar()[0]) + str(x.isocalendar()[1]).zfill(2)))
        return df['week_id']


def week_id_to_date(week_id):
    """
    Turn a Decathlon week id to date
    :param week_id: (int or pd.Series) the week id or pandas column of week ids
    :return: (pd.Timestamp or pd.Series) the date or pandas column of dates
    """
    assert isinstance(week_id, (int, np.integer, pd.Series, list))
    if isinstance(week_id, (int, np.integer)):
        assert is_iso_format(week_id), f"`week_id` {week_id} doesn't follow conform YYYYWW format"
        return pd.to_datetime(str(week_id) + '-0', format='%G%V-%w') - pd.Timedelta(1, unit='W')
    else:
        if isinstance(week_id, (list)):
            week_id = pd.Series(week_id)
        for w in week_id:
            assert is_iso_format(w), f"Week_id {w} doesn't follow conform YYYYWW format"
        return pd.to_datetime(week_id.astype(str) + '-0', format='%G%V-%w') - pd.Timedelta(1, unit='W')


def is_iso_format(week_id: int) -> bool:
    """
    Checks if `week` has conform YYYYWW ISO 8601 format.
    :param week_id: (int or pd.Series) the week id or pandas column of week ids
    :return: (bool) wether `week_id` follows the expected format
    """
    assert isinstance(week_id, (int, np.int32, np.int64)), "week_id must be of type integer."
    pattern = "^20[0-9]{2}(0[1-9]|[1-4][0-9]|5[0-3])$"
    is_iso_format = re.match(pattern, str(week_id))

    return is_iso_format is not None


def to_uri(bucket, key):
    """
    Transforms bucket & key strings into S3 URI
    :param bucket: (string) name of the S3 bucket
    :param key: (string) S3 key
    :return: (string) URI format
    """
    return f's3://{bucket}/{key}'


def from_uri(uri):
    """
    Transforms a S3 URI into bucket & key strings
    :param uri: (string) URI format
    :return bucket: (string) name of the S3 bucket
    :return key: (string) S3 key
    """
    o = urisplit(uri)
    bucket = o.authority
    key = o.path[1:]

    return bucket, key


def read_yml(file_path):
    """
    Read a local yaml file and return a python dictionary
    :param file_path: (string) full path to the yaml file
    :return: (dict) data loaded
    """

    if file_path[:2] == "s3":
        fs = s3fs.S3FileSystem()
        with fs.open(file_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)
    else:
        with open(file_path) as f:
            yaml_dict = yaml.safe_load(f)

    return yaml_dict


def read_csv_s3(bucket, file_path, header='infer', sep=',', parse_dates=False, names=None, usecols=None,
                compression='infer', escapechar=None):
    """
    Read a CSV file hosted on a S3 bucket, load and return as pandas dataframe
    :param bucket: (string) S3 source bucket
    :param file_path: (string) full path to the CSV file within this S3 bucket
    :param header: (string) optional, default is 'infer'
    :param sep: (string) optional, default is ';' (the separator char within the file)
    :param parse_dates: (list or bool) optional, default is False (names of date or datetime columns)
    :param names: (list) optional, default is None (use it to specify value for the read_csv underlying method called)
    :param usecols: (list) optional, default is None (use it to specify value for the read_csv underlying method called)
    :param compression: (string) optional, default is 'infer'
    :param escapechar: (string) optional, default is None (One-character string used to escape other characters)
    :return: (pandas DataFrame) data loaded
    """
    file_object = boto3.client('s3').get_object(Bucket=bucket, Key=file_path)
    file_body = io.BytesIO(file_object['Body'].read())

    data = pd.read_csv(file_body,
                       header=header,
                       sep=sep,
                       parse_dates=parse_dates,
                       names=names,
                       usecols=usecols,
                       compression=compression,
                       escapechar=escapechar)

    return data


def read_multipart_parquet_s3(bucket, dir_path, prefix_filename='part-'):
    """
    Read a multipart parquet file (splitted) hosted on a S3 bucket, load and return as pandas dataframe. Note that only
    files with name starting with <prefix_filename> value are taken into account.
    :param bucket: (string) S3 source bucket
    :param dir_path: (string) full path to the folder that contains all parts within this S3 bucket
    :param prefix_filename: (string) Optional. Default is 'part-' which is what Spark generates. Only files with name
    starting with this value will be loaded
    :return: (pandas DataFrame) data loaded
    """
    fs = s3fs.S3FileSystem()
    s3_uri = to_uri(bucket, dir_path)
    data = pq.ParquetDataset(s3_uri, filesystem=fs).read().to_pandas(date_as_object=False)

    return data


def write_str_to_file_on_s3(string, bucket, dir_path, verbose=False):
    """
    Write a string as a file on a S3 bucket.
    :param string: (str)
    :param bucket: (string) S3 source bucket
    :param dir_path: (string) full path to the folder that contains all parts within this S3 bucket
    starting with this value will be loaded
    :return: (pandas DataFrame) data loaded
    """
    resp = boto3.resource('s3').Object(bucket, dir_path).put(Body=string)

    if verbose:
        return resp


def write_df_to_parquet_on_s3(df, bucket, filename, index=False, verbose=False):
    """
    Write an in-memory pandas DataFrame to a parquet file on a S3 bucket
    :param dataframe: (pandas DataFrame) the data to save
    :param bucket: (string) S3 bucket name
    :param filename: (string) full path to the parquet file within the given bucket
    :param index: (bool) index value of the underlying 'to_parquet' function called
    """
    if verbose:
        print("Writing {} records to {}".format(len(df), to_uri(bucket, filename)))
    
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=index)
    boto3.resource('s3').Object(bucket, filename).put(Body=buffer.getvalue())
    

def import_modeling_parameters(environment: str) -> dict:
    """Handler to import modeling configuration from YML file

    Args:
        environment (str): Set of parameters on which to load the parameters

    Returns:
        A dictionary with all parameters for modeling import
    """
    params_full_path = os.path.join('config', f"{environment}.yml")
    assert os.path.isfile(params_full_path), f"Environment {environment} has no associated configuration file"

    params = read_yml(params_full_path)

    data_params = {'refined_data_global_bucket': params['buckets']['refined_data_global'],
                   'refined_data_specific_bucket': params['buckets']['refined_data_specific'],
                   'refined_global_path': params['paths']['refined_global_path'],
                   'refined_specific_path': params['paths']['refined_specific_path'],
                   'algorithm': params['modeling_parameters']['algorithm']
                   }

    return data_params


def check_environment(environment: str,
                      config_path: str = 'config/'
                      ) -> None:
    """
    Check if environment `environment` matches with a config file in path `config_path`

    Args:
        environment (str): environment config name to check for
        config_path (str): path in which to look for configuration files
    """

    assert isinstance(environment, (str)), "Variable `environment` must be a string"
    assert isinstance(config_path, (str)), "Variable `config_path` must be a string"

    if config_path[-1] != os.path.sep:
        config_path += os.path.sep
    assert os.path.exists(os.path.dirname(config_path)), f"Path {config_path} doesn't exist"

    regex = "^.*.yml$"
    rule = re.compile(regex)

    available_config = [c.replace('.yml', '') for c in os.listdir(config_path) if bool(rule.match(c))]

    assert environment in available_config, (f"Environment {environment} doesn't match with "
                                             "any configuration files available")


def get_current_week():
    """
    Return current week (international standard ISO 8601 - first day of week
    is Sunday, with format 'YYYYWW'
    :return current week (international standard ISO 8601) with format 'YYYYWW'
    """
    today = datetime.date.today()
    return date_to_week_id(today)


def check_list_cutoff(list_cutoff: Union[str, int, list]) -> list:
    """
    Check if `list_cutoff` is conform.

    Args:
        list_cutoff (str): list of cutoffs
    Return:
        conform_list_cutoff (str): list of cutoffs validated
    """

    assert isinstance(list_cutoff, (str, int, list)), ("list_cutoff must be a string representing "
                                                       "a list of cutoff, a list of integers or 'today'")

    if isinstance(list_cutoff, (str)):
        if list_cutoff == 'today':
            list_cutoff = [get_current_week()]
        else:
            list_cutoff = json.loads(list_cutoff)
            if isinstance(list_cutoff, (int)):
                list_cutoff = [list_cutoff]
    elif isinstance(list_cutoff, (int)):
        assert is_iso_format(list_cutoff), "Provided cutoff is not in ISO Format YYYYWW"
        list_cutoff = [list_cutoff]

    assert all([isinstance(cutoff, (int)) for cutoff in list_cutoff]), ("One of the provided cutoffs "
                                                                        "is not an integer")
    assert all([is_iso_format(cutoff) for cutoff in list_cutoff]), ("One of the provided cutoffs is not "
                                                                    "in iso format YYYYWW")

    return list_cutoff


def check_run_name(run_name: str) -> None:
    """
    Checks if `run_name` matches Sagemaker's regex on job_name.

    Args:
        run_name (str): Name to check the regez on.
    """
    assert isinstance(run_name, (str)), "Run_name should be a string"
    job_name_regex = "^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$"
    rule = re.compile(job_name_regex)

    assert rule.match(run_name), f"Run name {run_name} doesn't match Sagemaker Regex {job_name_regex}"

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
