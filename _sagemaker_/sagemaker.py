import json
import boto3
import yaml

def create_training_job(config):
    
    sm = boto3.client('sagemaker')
    resp = sm.create_training_job(
            TrainingJobName = config.get_train_job_name(),
            StoppingCondition = {
                                    'MaxRuntimeInSeconds': config.get_train_max_run_time()
                                },
            AlgorithmSpecification={
                'TrainingImage': config.get_train_docker_image(),
                'TrainingInputMode': "File"
            }, 
            RoleArn=config.get_global_role_arn(),
            InputDataConfig=[
                                {
                                    'ChannelName': 'training', # todo : check if it changes anything "ing"
                                    'DataSource': {
                                        'S3DataSource': {
                                            'S3DataType': 'S3Prefix',
                                            'S3Uri': 's3://' + config.get_train_bucket_input() + '/' + config.get_train_path_refined_data_input()
                                        }
                                    },
                                },
                            ], 
            OutputDataConfig={
                                'S3OutputPath': 's3://' + config.get_train_bucket_output() + '/' + config.get_train_path_refined_data_output()
                            },
            ResourceConfig={
                            'InstanceType': config.get_train_instance_type(),
                            'InstanceCount': config.get_train_instance_count(),
                            'VolumeSizeInGB': config.get_train_volume_size_in_gb()
                        },
            HyperParameters=config.get_train_hyperparameters(),
            Tags=[
                config.get_global_tags()
               ],
            VpcConfig={ 
                      'SecurityGroupIds': config.get_global_security_group_ids(),
                      'Subnets': config.get_global_subnets()
                    }
    )
    return resp

# todo : create hyperparameter tuning job