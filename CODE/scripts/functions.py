import boto3
import io
import json
import pandas as pd

def get_aws_account_id():
    """
    Get the AWS account ID.
    
    Returns:
    - AWS account ID as a string
    """
    sts_client = boto3.client('sts')
    response = sts_client.get_caller_identity()
    account_id = response['Account']
    return account_id

def read_settings(filename = "../SETTINGS/settings.json"):
    """
    Read JSON data from a file.
    
    Parameters:
    - filename: str, name of the file to read
    
    Returns:
    - data: dict, data read from the file
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data[get_aws_account_id()]


def save_df_to_s3(df, bucket_name, s3_key, decimal_places=2):
    """
    Save a DataFrame as a CSV file on S3 with specified decimal places.

    Parameters:
    - df: pd.DataFrame, the DataFrame to save
    - bucket_name: str, the S3 bucket name
    - s3_key: str, the S3 key (path) where the CSV will be saved
    - decimal_places: int, the number of decimal places to save in the CSV
    """
    # Convert DataFrame to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, float_format=f'%.{decimal_places}f')
    
    # Create S3 client
    s3_client = boto3.client('s3')
    
    # Upload CSV to S3
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())

# Function to delete all objects in an S3 folder
def delete_s3_folder(bucket_name, folder_path):
    """
    Delete all objects in an S3 folder.

    Parameters:
    - bucket_name: str, the S3 bucket name
    - folder_path: str, the S3 folder path
    """
    s3_client = boto3.client('s3')
    
    # List all objects in the folder
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
    
    # Check if there are any objects to delete
    if 'Contents' in response:
        objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
        
        # Delete the objects
        s3_client.delete_objects(Bucket=bucket_name, Delete={'Objects': objects_to_delete})
        
        print(f"Deleted {len(objects_to_delete)} objects from s3://{bucket_name}/{folder_path}")
    else:
        print(f"No objects found in s3://{bucket_name}/{folder_path}")

def read_csv_from_s3(bucket_name, s3_key):
    """
    Read a CSV file from S3 and return a DataFrame.

    Parameters:
    - bucket_name: str, the S3 bucket name
    - s3_key: str, the S3 key (path) to the CSV file

    Returns:
    - pd.DataFrame, the DataFrame read from the CSV file
    """
    # Create S3 client
    s3_client = boto3.client('s3')
    
    # Get the CSV file from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    csv_content = response['Body'].read().decode('utf-8')
    
    # Read the CSV content into a DataFrame
    df = pd.read_csv(io.StringIO(csv_content))
    
    return df