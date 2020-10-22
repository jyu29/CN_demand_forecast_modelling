import json
import s3fs
import datetime
import numpy as np
import pandas as pd


def read_json(file_path):
    """
    Read a local JSON file and return a python dictionary
    :param file_path: (string) full path to the JSON file
    :return: (dict) data loaded
    """

    if file_path[:2] == "s3":
        fs = s3fs.S3FileSystem()
        with fs.open(file_path, 'r') as f:
            json_dict = json.load(f)
    else:
        with open(file_path) as f:
            json_dict = json.load(f)

    return json_dict


def pretty_print_json(json_str):
    """
    Pretty prints a json
    :param json: (str), json in 'str' format
    """

    print(json.dumps(json_str, indent=4, sort_keys=True))


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
