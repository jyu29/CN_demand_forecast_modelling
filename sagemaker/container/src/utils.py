import boto3
import pandas as pd
import numpy as np
import isoweek
import io


def get_next_week_id(week_id):
    """
    ARGUMENTS:
    
    date ( integer ): week identifier in the format 'year'+'week_number'
    
    RETURNS:
    
    next week in the same format as the date argument
    """
    if not(isinstance(week_id, (int, np.integer))):
        return 'DATE ARGUMENT NOT AN INT'
    if len(str(week_id)) != 6:
        return 'UNVALID DATE FORMAT'

    year = week_id // 100
    week = week_id % 100

    if week < 52:
        return week_id + 1
    elif week == 52:
        last_week = isoweek.Week.last_week_of_year(year).week
        if last_week == 52:
            return (week_id // 100 + 1) * 100 + 1
        elif last_week == 53:
            return week_id + 1
        else:
            return 'UNVALID ISOWEEK.LASTWEEK NUMBER'
    elif week == 53:
        if isoweek.Week.last_week_of_year(year).week == 52:
            return 'UNVALID WEEK NUMBER'
        else:
            return (date // 100 + 1) * 100 + 1
    else:
        return 'UNVALID DATE'


def get_next_n_week(week_id, n):
    next_n_week = [week_id]
    for i in range(n-1):
        week_id = get_next_week_id(week_id)
        next_n_week.append(week_id)
    return next_n_week


def week_id_to_date(week_id):
    assert isinstance(week_id, (int, np.integer, pd.Series))
    
    if isinstance(week_id, (int, np.integer)):
        return pd.to_datetime(str(week_id) + '-0', format='%G%V-%w') - pd.Timedelta(1, unit='W')
    else:
        return pd.to_datetime(week_id.astype(str) + '-0', format='%G%V-%w') - pd.Timedelta(1, unit='W')
    

def download_file_from_S3(bucket, s3_file_path, local_path):
    # Any clients created from this session will use credentials
    # from the [profile_name] section of ~/.aws/credentials.

    s3client = boto3.client('s3')
    s3client.download_file(bucket, s3_file_path, local_path)

    return print(
        "downloaded " +
        bucket +
        "/" +
        s3_file_path +
        " to: " +
        local_path)


def write_csv_S3(df, bucket, file_path, sep = '|', compression=None):
    
    # /!\ only works with 'gzip' or None compression atm
    assert compression in ["gzip", None]
    
    s3client = boto3.client("s3")
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, sep=sep)
    csv_buffer.seek(0)
        
    if compression == None:
        s3client.put_object(Bucket=bucket, Key=file_path, Body=csv_buffer.getvalue())
        
    if compression == "gzip":
        gz_buffer = io.BytesIO()
        with gzip.GzipFile(mode="w", fileobj=gz_buffer) as gz_file:
            gz_file.write(bytes(csv_buffer.getvalue(), 'utf-8'))
        s3client.put_object(Bucket=bucket, Key=file_path, Body=gz_buffer.getvalue())

    print('>> Data written on s3://' + bucket + '/' + file_path)