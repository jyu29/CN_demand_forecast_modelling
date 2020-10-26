import s3fs
import datetime
import yaml
import pprint
import io
import boto3
import numpy as np
import pandas as pd


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
    :param dict: python dictionary
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
    assert isinstance(week_id, (int, np.integer, pd.Series))
    if isinstance(week_id, (int, np.integer)):
        return pd.to_datetime(str(week_id) + '-0', format='%G%V-%w') - pd.Timedelta(1, unit='W')
    else:
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