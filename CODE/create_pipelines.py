import boto3
import sagemaker
import sagemaker.session
import os
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import TrainingStep
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep
from sagemaker.inputs import TransformInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker import get_execution_role
from sagemaker.workflow.parameters import ParameterString, ParameterInteger

from scripts.functions import *

import os
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

if __name__ == '__main__':
    settings = read_settings('scripts/settings.json')
    
    region = boto3.Session().region_name
    sagemaker_session = sagemaker.session.Session()
    role = sagemaker.get_execution_role()
    default_bucket = sagemaker_session.default_bucket()

    pipeline_session = PipelineSession()

    # Deployment pipeline
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type='ml.t3.medium',
        instance_count=1,
        base_job_name=settings['preprocessing_job_name'],
        sagemaker_session=pipeline_session,
        role=role,
        env=settings
    )
    
    train_s3_path = os.path.join("s3://",settings['bucket_name'],settings['project_path_s3'],'data','train')
    test_s3_path = os.path.join("s3://",settings['bucket_name'],settings['project_path_s3'],'data','test')
    valid_s3_path = os.path.join("s3://",settings['bucket_name'],settings['project_path_s3'],'data','valid')
    inference_train_s3_path = os.path.join("s3://",settings['bucket_name'],settings['project_path_s3'],'data','inference_train')
    
    processor_args = sklearn_processor.run(
        inputs=[],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source=settings["preprocessing_output_train"],
                destination=train_s3_path),
            ProcessingOutput(
                output_name="test",
                source=settings["preprocessing_output_test"],
                destination=test_s3_path),
            ProcessingOutput(
                output_name="valid",
                source=settings["preprocessing_output_valid"],
                destination=valid_s3_path),
            ProcessingOutput(
                output_name="inference_train",
                source=settings["preprocessing_output_inference_train"],
                destination=inference_train_s3_path)        
    
        ],
        code="scripts/processing.py",
    ) 
    
    step_process = ProcessingStep(
        name=settings["preprocessing_step_name"],
        step_args=processor_args
    )
    
    environment = {
        'mlflow_arn': settings['mlflow_arn'],
        'mlflow_experiment_name': settings['mlflow_experiment_name'],
        'mlflow_final_model_name': 'final-model2',
        'mlflow_model_name': settings['mlflow_model_name']
    }
    
    sklearn = SKLearn(
        entry_point='train.py', # The file with the training code
        source_dir='scripts', # The folder with the training code
        framework_version='1.2-1', # Version of SKLearn which will be used
        instance_type='ml.m5.large', # Instance type that wil be used
        role=role, # Role that will be used during execution
        sagemaker_session=pipeline_session, 
        base_job_name=settings['training_job_name'], # Name of the training job. Timestamp will be added as suffix
        environment = environment
    )
    
    train_args = sklearn.fit({"train": step_process.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri})
    
    step_train = TrainingStep(
        name=settings["training_step_name"],
        step_args = train_args
    )
    
    mlflow.set_tracking_uri(settings['mlflow_arn'])
    mlflow.set_experiment(settings['mlflow_experiment_name'])
    client = MlflowClient()
    role = get_execution_role()
    
    registered_model = client.get_registered_model(name=settings['mlflow_model_name'])
    run_id = registered_model.latest_versions[0].run_id
    source_path = registered_model.latest_versions[0].source
    model_path = os.path.join(source_path, 'model.tar.gz')
    
    # Create the SKLearnModel
    sklearn_model = SKLearnModel(
        model_data=model_path,
        entry_point='inference.py', # The file with the training code
        source_dir="scripts", # The folder with the training code
        role=role,
        framework_version='1.2-1',  # Replace with the appropriate sklearn version
        sagemaker_session=pipeline_session
    )
    
    step_create_model = ModelStep(
        name=settings["modelcreate_step_name"],
        step_args=sklearn_model.create(instance_type="ml.m5.large"),
        depends_on=[step_train]
    )
    
    transformer_output_path = os.path.join("s3://",settings['bucket_name'],settings['project_path_s3'],'output','inference_train')
    
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=transformer_output_path,
        accept="text/csv"
    )
    
    step_transform = TransformStep(
        name=settings["transformer_step_name"],
        transformer=transformer,
        inputs=TransformInput(
            data=step_process.properties.ProcessingOutputConfig.Outputs['inference_train'].S3Output.S3Uri,
            content_type='text/csv', # It is neccessary because csv is not default format
            split_type='Line' # Each line equals one observation)
    ))
    
    verifypredictions_output_s3 = os.path.join("s3://",settings['bucket_name'],settings['project_path_s3'],'output','verify')
    
    processor_args = sklearn_processor.run(
        inputs=[],
        outputs=[
            ProcessingOutput(
                output_name="default",
                source=settings["verifypredictions_output_default"],
                destination=verifypredictions_output_s3)
        ],
        code="scripts/verify_predictions.py",
    ) 
    
    step_verify = ProcessingStep(
        name=settings["verifypredictions_step_name"],
        step_args=processor_args,
        depends_on=[step_transform],
    )
    
    pipeline_name = f"01-churn-deploy-model1"
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_process, step_train, step_create_model, step_transform, step_verify]    
    )
    
    pipeline.upsert(role_arn=role)

    # Inference pipeline
    input_path = ParameterString(name="InputPath", default_value="")
    output_path = ParameterString(name="OutputPath", default_value="")
    
    transformer_i = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=output_path,
        accept="text/csv"
    )
    
    step_transform_i = TransformStep(
        name="transformer_step_name",
        transformer=transformer_i,
        inputs=TransformInput(
            data=input_path,
            content_type='text/csv', # It is necessary because csv is not default format
            split_type='Line' # Each line equals one observation
        )
    )
    
    pipeline_name_i = "01-churn-inference"
    pipeline_i = Pipeline(
        name=pipeline_name_i,
        steps=[step_transform_i],
        parameters=[input_path, output_path]
    )
    
    pipeline_i.upsert(role_arn=role)