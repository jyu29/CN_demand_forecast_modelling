"""
Python script to orchestrate demand forecast modeling:
- Engineers features & prepares data in DeepAR format
- Creates Training Docker Image
- Pops instance to  preprocess train a DeepAR model, then output predictions
@author: benbouillet ( Benjamin Bouillet )
"""
import argparse
import json

import src.sagemaker_utils as su
import src.utils as ut


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', choices=['dev', 'prod', 'dev_old'], default="dev",
                        help="'dev' or 'prod', to set the right configurations")
    parser.add_argument('--list_cutoff', default=str('today'), help="List of cutoffs in format YYYYWW between brackets or 'today'")
    parser.add_argument('--run_name', help="Run Name for file hierarchy purpose")
    args = parser.parse_args()

    # Defining variables
    environment = args.environment
    if args.list_cutoff == 'today':
        list_cutoff = [ut.get_current_week()]
    else:
        list_cutoff = json.loads(args.list_cutoff)

    for cutoff in list_cutoff:
        assert type(cutoff) == int
    
    assert type(args.run_name) == str
    run_name = args.run_name

    # import parameters
    params_full_path = f"config/{environment}.yml"
    params = ut.read_yml(params_full_path)

    # Getting variables for code readability below
    algo = params['functional_parameters']['algorithm']

    # Building custom full paths
    refined_specific_path = params['paths']['refined_specific_path']
    params['paths']['refined_specific_path_full'] = f"{refined_specific_path}{run_name}/{algo}/"

    print(f"Starting modeling run '{run_name}' for cutoff {list_cutoff} in {environment} environment with parameters:")
    ut.pretty_print_dict(params)

    # SAGEMAKER #
    sm_handler = su.SagemakerHandler(run_name, list_cutoff, params)
    # Generating df_jobs
    sm_handler.generate_df_jobs()
    # Feature generation
    sm_handler.generate_input_data_all_cutoffs()
    # Training Job
    sm_handler.launch_training_jobs()
    # Transform job
    sm_handler.launch_transform_jobs()
