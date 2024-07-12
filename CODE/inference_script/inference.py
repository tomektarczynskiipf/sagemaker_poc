
import os
import pickle
import json
import pandas as pd
import io
from sagemaker_containers.beta.framework import worker

def model_fn(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(input_data, content_type):
    if content_type == 'text/csv':
        return pd.read_csv(io.StringIO(input_data))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    predictions = model.predict_proba(input_data)
    return predictions

def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return worker.Response(json.dumps(prediction.tolist()), mimetype=content_type)
    elif content_type == 'text/csv':
        return worker.Response(pd.DataFrame(prediction).to_csv(index=False), mimetype=content_type)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
