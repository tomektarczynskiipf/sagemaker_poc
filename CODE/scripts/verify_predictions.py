import os
import pandas as pd
import numpy as np
import boto3
from io import StringIO
from urllib.parse import urlparse

def read_csv_from_s3(s3_path):
    # Parse the S3 path
    print(f"Reading dataframe from following S3 path: {s3_path}")
    
    parsed_url = urlparse(s3_path)
    bucket_name = parsed_url.netloc
    file_key = parsed_url.path.lstrip('/')

    # Initialize a session using Amazon S3
    s3_client = boto3.client('s3')

    # Get the CSV file content from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')

    # Read the CSV content into a pandas DataFrame
    df = pd.read_csv(StringIO(csv_content))

    return df

if __name__ == '__main__':
    settings = os.environ
    
    df = read_csv_from_s3(settings["reference_inference_train"])
    
    df_base = read_csv_from_s3(settings["reference_inference_train"])
    df_base.columns = ['pred_base', 'target']
    
    print("DataFrame base head:\n", df_base.head())
    
    pipeline_prediction_path = os.path.join("s3://",settings['bucket_name'],settings['project_path_s3'],'output','inference_train', 'inference_train.csv.out')
    
    df_pipeline = read_csv_from_s3(pipeline_prediction_path)
    df_pipeline.columns = ['pred_pipeline']
    print("DataFrame pipeline head:\n", df_pipeline.head())
    
    df_all = pd.concat([df_base, df_pipeline], axis = 1)
    df_all['abs_diff'] = abs(df_all['pred_base'] - df_all['pred_pipeline'])
    print("DataFrame all head:\n", df_all.head())
    
    max_diff = df_all['abs_diff'].max()
    print(f"Maximum difference in abs_diff column: {max_diff}")
    
    threshold = float(settings["max_diff_pred_train_accept"])
    print(f"Maximum threshold for difference in predictions is: {threshold}")
    
    if max_diff > threshold:
        raise ValueError(f"Maximum difference {max_diff} exceeds the acceptable threshold of {threshold}")
    else:
        print("Maximum difference is within the acceptable threshold.")