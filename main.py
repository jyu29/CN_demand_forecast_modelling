"""
Python script to orchestrate demand forecast modeling:
- Engineers features & prepares data in DeepAR format
- Creates Training Docker Image
- Pops instance to  preprocess train a DeepAR model, then output predictions
@author: benbouillet ( Benjamin Bouillet )
"""
import argparse
import json

import s3fs
import src.sagemaker_utils as su
import src.utils as ut

fs = s3fs.S3FileSystem()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', choices=['dev', 'prod', 'dev_old'], default="dev",
                        help="'dev' or 'prod', to set the right configurations")
    parser.add_argument('--list_cutoff', default=str([ut.get_current_week()]), help="List of cutoffs in format YYYYWW between brackets or 'today'")
    args = parser.parse_args()

    # Defining variables
    environment = args.environment
    if args.list_cutoff == 'today':
        list_cutoff = [ut.get_current_week()]
    else:
        list_cutoff = json.loads(args.list_cutoff)

    for cutoff in list_cutoff:
        assert type(cutoff) == int

    # import parameters
    params_full_path = f"config/{environment}.yml"
    params = ut.read_yml(params_full_path)

    # Getting variables for code readability below
    run_name = params['functional_parameters']['run_name']
    algo = params['functional_parameters']['algorithm']

    # Building custom full paths
    refined_specific_path = params['paths']['refined_specific_path']
    params['paths']['refined_specific_path_full'] = f"{refined_specific_path}{run_name}/{algo}/"

    print(f"Starting modeling for cutoff {list_cutoff} in {environment} environment with parameters:")
    ut.pretty_print_dict(params)

    # Monitoring DataFrame creation
    df_jobs = su.generate_df_jobs(run_name,
                                  list_cutoff,
                                  params['buckets']['refined-data'],
                                  f"{params['paths']['refined_specific_path_full']}"
                                  )

    # Feature generation
    #df_jobs.apply(lambda row: su.generate_input_data(row, fs, params), axis=1)

    # SAGEMAKER #
    sm_handler = su.SagemakerHandler(df_jobs, params)
    # Training Job
    sm_handler.launch_training_jobs()
    # Transform job
    sm_handler.launch_transform_jobs()
