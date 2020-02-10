"""
Python script to orcherstrate demand forecast modeling:
- Creates Training Docker Image
- Pops instance to  preprocess ( reformat ) model input data, train a fprophet model, then output predictions
@author: oaitelkadi ( Ouiame Ait El Kadi )
"""
import argparse
import boto3
import os
import subprocess
import time

import src.config as cf
import _sagemaker_.sagemaker as sg


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', choices=['dev','prod'], default="dev",
                        help="'dev' or 'prod', to set the right configurations")
    parser.add_argument('--only_last', choices=['True','False'], default='True',
                        help='Run only for last week?')
    args = parser.parse_args()
    
    config_file = "conf/prod.yml" if args.environment=="prod" else "conf/dev.yml"
    config = cf.ProgramConfiguration(config_file, "conf/functional.yml")
    
    # Create Docker image for training
    print("Building Docker Image...")
    subprocess.call(['sh', '_sagemaker_/build_image.sh', config.get_train_image_name(), args.environment, args.only_last])
    
    # Create a SageMaker training job through an API call
    print("Creating Training Job...")
    sg_resp = sg.fct_create_training_job(config)  # todo : david suggestion - add env variables to hyperparameters, checkout create_algorithm()

    # Monitor the status of the launched training job
    print("Monitoring training job status...")
    client = boto3.client('sagemaker')
    while True:
        status = client.describe_training_job(TrainingJobName=sg_resp["TrainingJobArn"].split("/")[-1])['SecondaryStatus']
        print(status)
        if status in ['Starting', 'LaunchingMLInstances', 'PreparingTrainingStack', 'Downloading', 'DownloadingTrainingImage', 'Training', 'Uploading']:
            time.sleep(config.get_monitor_sleep())
        elif status == 'Completed':
            break
        else:
            raise Exception('Training job has failed !')
