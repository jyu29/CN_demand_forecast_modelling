import json
import boto3
import yaml

def create_training_job(config):
    
    sm = boto3.client('sagemaker')
    resp = sm.create_training_job(
            TrainingJobName = config.get_train_job_name(), 
            AlgorithmSpecification={
                'TrainingImage': config.get_train_docker_image(),
            }, 
            RoleArn=config.get_global_role_arn(),
            InputDataConfig=[
                                {
                                    'ChannelName': 'train',
                                    'DataSource': {
                                        'S3DataSource': {
                                            'S3Uri': 's3://' + config.get_train_bucket_input() + '/' + \
                                                      config.get_train_path_refined_data_input()
                                        }
                                    },
                                },
                            ], 
            OutputDataConfig={
                                'S3OutputPath': config.get_train_path_refined_data_output()
                            },
            ResourceConfig={
                            'InstanceType': config.get_train_instance_type(),
                            'InstanceCount': config.get_train_instance_count()
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