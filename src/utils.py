import datetime
import gzip
import io
import pprint
import re

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
import yaml
from uritools import urisplit


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


def pretty_print_dict(dict_to_print):
    """
    Pretty prints a dictionary
    :param dict_to_print: python dictionary
    """

    pprint.pprint(dict_to_print)


def get_current_week():
    """
    Return current week (international standard ISO 8601 - first day of week
    is Sunday, with format 'YYYYWW'
    :return current week (international standard ISO 8601) with format 'YYYYWW'
    """
    today = datetime.date.today()
    return date_to_week_id(today)


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
    pattern = "^20[0-9]{2}(0[1-9]|[1-4][0-9]|5[0-3])$"
    if isinstance(week_id, (int, np.integer)):
        assert re.match(pattern, str(week_id)), f"`week_id` {week_id} doesn't follow conform YYYYWW format"
        return pd.to_datetime(str(week_id) + '-0', format='%G%V-%w') - pd.Timedelta(1, unit='W')
    else:
        if isinstance(week_id, (list)):
            week_id = pd.Series(week_id)
        for w in week_id:
            assert re.match(pattern, str(w)), f"Week_id {w} doesn't follow conform YYYYWW format"
        return pd.to_datetime(week_id.astype(str) + '-0', format='%G%V-%w') - pd.Timedelta(1, unit='W')


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


def write_df_to_csv_on_s3(df, bucket, filename, sep=',', header=True, index=False, compression=None, verbose=True):
    """
    Write an in-memory pandas DataFrame to a CSV file on a S3 bucket
    :param df: (pandas DataFrame) the data to save
    :param bucket: (string) S3 bucket name
    :param filename: (string) full path to the CSV file within the given bucket
    :param sep: (string) separator char to use
    :param header: (bool or list of str) header value of the underlying 'to_csv' function called
    :param index: (bool) index value of the underlying 'to_csv' function called
    :param compression: (string) Optional. Default is None. If set to "gzip" then file is written in a compressed .gz
    format
    :param verbose: (bool) Optional. Default is True. If True, will display the writing path.
    """
    if verbose:
        print("Writing {} records to {}".format(len(df), to_uri(bucket, filename)))
    # Create buffer
    csv_buffer = io.StringIO()
    # Write dataframe to buffer
    df.to_csv(csv_buffer, sep=sep, header=header, index=index)
    buffer = csv_buffer

    # Handle potential compression
    if compression == "gzip":
        gz_buffer = io.BytesIO()
        with gzip.GzipFile(mode="w", fileobj=gz_buffer) as gz_file:
            gz_file.write(bytes(csv_buffer.getvalue(), 'utf-8'))
        buffer = gz_buffer

    # Write buffer to S3 object
    boto3.resource('s3').Object(bucket, filename).put(Body=buffer.getvalue())


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


def import_raw_config(environment: str) -> dict:
    """Handler to import full configuration from YML file

    Args:
        environment (str):

    Returns:
        A dictionary with all parameters
    """
    params_full_path = f"config/{environment}.yml"
    params = read_yml(params_full_path)

    return params

def import_refining_config(environment: str,
                           cutoff: int,
                           run_name: str,
                           train_path: str,
                           predict_path: str
                           ) -> dict:
    """Handler to import specific refining configuration from YML file

    Args:
        environment (str):
        cutoff (int):
        run_name (str):
        df_jobs (pd.DataFrame):

    Returns:
        A dictionary with all parameters for specific refining process
    """
    params_full_path = f"config/{environment}.yml"
    params = read_yml(params_full_path)

    refining_params = {'cutoff': cutoff,
                       'rec_cold_start_length': params['refining_specific_parameters']['rec_cold_start_length'],
                       'prediction_length': params['modeling_parameters']['hyperparameters']['prediction_length'],
                       'patch_covid': params['refining_specific_parameters']['patch_covid'],
                       'patch_covid_weeks': params['refining_specific_parameters']['patch_covid_weeks'],
                       'rec_cold_start_group': params['refining_specific_parameters']['rec_cold_start_group'],
                       'refined_global_bucket': params['buckets']['refined_data_global'],
                       'refined_specific_bucket': params['buckets']['refined_data_specific'],
                       'output_paths': {'train_path': train_path,
                                        'predict_path': predict_path
                                        }
                       }
    return refining_params


def import_sagemaker_params(environment: str,
                            ) -> dict:
    """Handler to import sagemaker configuration from YML file

    Args:
        environment (str):
        cutoff (int):
        run_name (str):

    Returns:
        A dictionary with all parameters for sagemaker training & inference
    """
    params_full_path = f"config/{environment}.yml"
    params = read_yml(params_full_path)

    sagemaker_params = {'bucket': params['buckets']['refined_data_specific'],
                        'refined_path': params['paths']['refined_specific_path'],
                        'train_instance_type': params['modeling_parameters']['train_instance_type'],
                        'train_instance_count': params['modeling_parameters']['train_instance_count'],
                        'train_max_instances': params['modeling_parameters']['train_max_instances'],
                        'train_use_spot_instances': params['modeling_parameters']['train_use_spot_instances'],
                        'transform_instance_type': params['modeling_parameters']['transform_instance_type'],
                        'transform_instance_count': params['modeling_parameters']['transform_instance_count'],
                        'transform_max_instances': params['modeling_parameters']['transform_max_instances'],
                        'role': params['modeling_parameters']['role'],
                        'image_name_label': params['modeling_parameters']['image_name_label'],
                        'tags': [params['modeling_parameters']['tags']],
                        'hyperparameters': params['modeling_parameters']['hyperparameters']
                        }

    return sagemaker_params
