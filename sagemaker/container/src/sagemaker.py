import json
import boto3
import time



def create_training_job(config):
   
    sm = boto3.client('sagemaker')
    resp = sm.create_training_job(
            TrainingJobName = config.get_train_job_name(), 
            AlgorithmSpecification={
                'TrainingInputMode': 'File',
                'TrainingImage': '783357654285.dkr.ecr.<your-region>.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3',
            }, 
            RoleArn=role_arn,
            InputDataConfig=[
                                {
                                    'ChannelName': 'train',
                                    'DataSource': {
                                        'S3DataSource': {
                                            'S3DataType': 'S3Prefix',
                                            'S3Uri': 's3://<path-to-training-data-dir>/',
                                            'S3DataDistributionType': 'FullyReplicated',
                                        }
                                    },
                                },
                            ], 
            OutputDataConfig={
                                'S3OutputPath': 's3://<path-to-output_dir>/'
                            },
            ResourceConfig={
                            'InstanceType': 'ml.m4.xlarge',
                            'InstanceCount': 1,
                            'VolumeSizeInGB': 30,
                        }, 
            StoppingCondition={
                                'MaxRuntimeInSeconds': 600
                            },
            HyperParameters={
                'sagemaker_submit_directory': 's3://<path-to-sagemaker_submit_directory>/',
                'sagemaker_program' : "train_and_deploy.py",
                'sagemaker_region': "<your-region>",
                'sagemaker_job_name': training_job_name,
                'sagemaker_submit_directory': src_path
            },
            Tags=[]
    )
    
    
    
def create_hyperparameter_tuning_job(training_job_name):
    sm = boto3.client('sagemaker')
    model = sm.create_model(
        ModelName=training_job_name,
        PrimaryContainer={
            'ContainerHostname': 'model-Container',
            'Image': '783357654285.dkr.ecr.<your-region>.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3',
            'ModelDataUrl': f's3://<path-to-training-job-output_dir>/{training_job_name}/output/model.tar.gz',
            'Environment': {
                'SAGEMAKER_PROGRAM': 'train_and_deploy.py',
                'SAGEMAKER_REGION':'<your-region>',
                'SAGEMAKER_SUBMIT_DIRECTORY': src_path
    
            },
        },
        ExecutionRoleArn=role_arn,
    )
    endpoint_config = sm.create_endpoint_config(
        EndpointConfigName=training_job_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': training_job_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.t2.medium',
            },
        ],
    )
    
    sm.update_endpoint(EndpointName=endpoint_name,
                   EndpointConfigName=training_job_name)
