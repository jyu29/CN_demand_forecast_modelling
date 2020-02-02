"""
Python script to orcherstrate demand forecast modeling:
- Creates Training Docker Image
- Preprocesses ( reformats ) model input data
- Pops instance to train
@author: oaitelkadi ( Ouiame Ait El Kadi )
"""
import argparse
import subprocess
import os

import src.config as cf
#import src.preprocess as pp
import _sagemaker_.sagemaker as sg


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', choices=['prepord','prod'], default="preprod",
                        help="'preprod' or 'prod', to set the right configurations")
    parser.add_argument('--only_last', default='True',
                        help='Run only for last week?')
    args = parser.parse_args()
    
    config_file = "conf/prod.yml" if args.environment=="prod" else "conf/dev.yml"
    config = cf.ProgramConfiguration(config_file, "conf/functional.yml")
    
    #os.environ["RUN_ENV"] = args.environment
    #os.environ["ONLY_LAST"] = args.only_last
    
    # Create Docker image for training
    print("Building Docker Image...")
    subprocess.call(['sh', '_sagemaker_/build_image.sh', config.get_train_image_name(), args.environment, args.only_last])
    #os.system('sh _sagemaker_/build_image.sh ' + config.get_train_image_name())
    
    # Preprocessing file
    #pp.format_cutoff_train_data(config, only_last=eval(args.only_last))
    
    # Train model
    print("Creating Training Job...")
    sg.create_training_job(config)