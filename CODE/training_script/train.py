
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
    model_dir = os.environ['SM_MODEL_DIR'] # Folder where model must be saved
    train_dir = os.environ['SM_CHANNEL_TRAIN'] # Folder where train data is stored
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']

    # Lets assume there is only one training file
    train_file_name = os.listdir(train_dir)[0]
    train_file_path = os.path.join(train_dir, train_file_name)
    
    train_data = pd.read_csv(train_file_path, engine="python")

    # labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]  

    mlflow.set_tracking_uri(os.environ['mlflow_arn'])
    mlflow.set_experiment(os.environ['mlflow_experiment_name'])
    
    with mlflow.start_run(run_name = os.environ['mlflow_final_model_name']):
        mlflow.set_tag("training_job_name", os.environ['TRAINING_JOB_NAME'])
        
        clf = LogisticRegression(max_iter=100)
        clf = clf.fit(train_X, train_y)
    
        mlflow.sklearn.log_model(clf, "model")

        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(clf, model_path)
    

        # Pack model.pkl into model.tar.gz
        tar_path = os.path.join(output_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_path, arcname=os.path.basename(model_path))

        # Log the model.tar.gz as an artifact in MLflow
        mlflow.log_artifact(tar_path, artifact_path="model")
        os.remove(tar_path)

    # Register the model with MLflow
    run_id = mlflow.last_active_run().info.run_id
    artifact_path = "model"
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
    model_details = mlflow.register_model(model_uri=model_uri, name=os.environ['mlflow_model_name'])
