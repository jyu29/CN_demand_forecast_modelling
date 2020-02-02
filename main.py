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
import src.preprocess as pp
import sagemaker.sagemaker as sg


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', required=False, choices=['prepord','prod'], default="preprod",
                        help="'preprod' or 'prod', to set the right configurations")
    parser.add_argument('--only_last', default='True',
                        help='Run only for last week?')
    args = parser.parse_args()
    
    config_file = "config/prod.yaml" if args.environment=="prod" else "config/dev.yaml"
    config = cf.ProgramConfiguration(config_file, "config/functional.yaml")
    
    os.environ["RUN_ENV"] = args.environment
    os.environ["ONLY_LAST"] = args.only_last
    
    # Create Docker image for training
    subprocess.call(['sh', 'sagemaker/build_image.sh', config.get_image_name()])
    
    # Preprocessing file
    #pp.format_cutoff_train_data(config, only_last=eval(args.only_last))
    
    # Train model
    sg.create_training_job(config)