import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep
from sagemaker.inputs import TransformInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString

if __name__ == '__main__':
    role = sagemaker.get_execution_role()
    pipeline_session = PipelineSession()
    
    input_path = ParameterString(name="InputPath", default_value="")
    output_path = ParameterString(name="OutputPath", default_value="")
    
    transformer = Transformer(
        model_name="pipelines-6i7hk20pbxwo-01-churn-model-creat-XupKjmb2SN",
        instance_type="ml.m5.large",
        instance_count=1,
        output_path=output_path,
        accept="text/csv"
    )
    
    step_transform = TransformStep(
        name="transformer_step_name",
        transformer=transformer,
        inputs=TransformInput(
            data=input_path,
            content_type='text/csv', # It is necessary because csv is not default format
            split_type='Line' # Each line equals one observation
        )
    )
    
    pipeline_name = "01-churn-inference"
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_transform],
        parameters=[input_path, output_path]
    )
    
    pipeline.upsert(role_arn=role)