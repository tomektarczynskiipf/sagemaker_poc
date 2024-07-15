
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
import tarfile
from sklearn.linear_model import LogisticRegression
import mlflow
import shutil

if __name__ == '__main__':
    print("Starting script...")
    
    model_dir = os.environ['SM_MODEL_DIR'] # Folder where model must be saved
    print(f"Model directory: {model_dir}")
    
    train_dir = os.environ['SM_CHANNEL_TRAIN'] # Folder where train data is stored
    print(f"Training data directory: {train_dir}")
    
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    print(f"Output data directory: {output_dir}")

    # Lets assume there is only one training file
    train_file_name = os.listdir(train_dir)[0]
    print(f"Training file name: {train_file_name}")
    
    train_file_path = os.path.join(train_dir, train_file_name)
    print(f"Training file path: {train_file_path}")
    
    train_data = pd.read_csv(train_file_path, engine="python")
    print("Training data loaded successfully")

    # Labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]  
    print("Training data split into features and labels")

    mlflow.set_tracking_uri(os.environ['mlflow_arn'])
    print(f"MLflow tracking URI set to: {os.environ['mlflow_arn']}")
    
    mlflow.set_experiment(os.environ['mlflow_experiment_name'])
    print(f"MLflow experiment set to: {os.environ['mlflow_experiment_name']}")
    
    with mlflow.start_run(run_name=os.environ['mlflow_final_model_name']):
        print(f"Started MLflow run with name: {os.environ['mlflow_final_model_name']}")
        
        mlflow.set_tag("training_job_name", os.environ['TRAINING_JOB_NAME'])
        print(f"Set MLflow tag: training_job_name={os.environ['TRAINING_JOB_NAME']}")
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        print("Initialized LogisticRegression model")
        
        clf = clf.fit(train_X, train_y)
        print("Model training completed")
    
        mlflow.sklearn.log_model(clf, "model")
        print("Model logged in MLflow")

        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(clf, model_path)
        print(f"Model saved to {model_path}")

        # Pack model.pkl into model.tar.gz
        tar_path = os.path.join(output_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_path, arcname=os.path.basename(model_path))
        print(f"Model tar.gz archive created at {tar_path}")

        # Log the model.tar.gz as an artifact in MLflow
        mlflow.log_artifact(tar_path, artifact_path="model")
        print("Model tar.gz logged as artifact in MLflow")
        
        os.remove(tar_path)
        print(f"Temporary tar.gz file removed: {tar_path}")

    # Register the model with MLflow
    run_id = mlflow.last_active_run().info.run_id
    artifact_path = "model"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_details = mlflow.register_model(model_uri=model_uri, name=os.environ['mlflow_model_name'])
    print(f"Model registered in MLflow with URI: {model_uri} and name: {os.environ['mlflow_model_name']}")
