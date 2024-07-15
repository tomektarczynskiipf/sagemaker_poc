
import os
import pickle
import json
import pandas as pd
import io
from sagemaker_containers.beta.framework import worker
import joblib

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model

def input_fn(input_data, content_type):
    if content_type == 'text/csv':
        return pd.read_csv(io.StringIO(input_data))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    # Ensure the input data is in the correct format
    if isinstance(input_data, pd.DataFrame):
        predictions = model.predict_proba(input_data)[:, 1]
    else:
        raise ValueError("Input data should be a pandas DataFrame")
    return predictions

def output_fn(prediction, accept='application/json'):
    if accept == 'application/json':
        return json.dumps(prediction.tolist())
    elif accept == 'text/csv':
        return pd.DataFrame(prediction).to_csv(index=False)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
