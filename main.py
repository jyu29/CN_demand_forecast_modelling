"""
Python script to orchestrate demand forecast modeling:
- Engineers features & prepares data in DeepAR format
- Creates Training Docker Image
- Pops instance to  preprocess train a DeepAR model, then output predictions
@author: benbouillet ( Benjamin Bouillet )
"""
import s3fs
import src.utils as ut
import src.sagemaker_utils as su
import argparse

fs = s3fs.S3FileSystem()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', choices=['dev', 'prod'], default="dev",
                        help="'dev' or 'prod', to set the right configurations")
    parser.add_argument('--cutoff', default=ut.get_current_week(), help="cutoff in format YYYYWW or 'today'")
    args = parser.parse_args()

    # Defining variables
    environment = args.environment
    cutoff = args.cutoff
    run_name = f"{environment}-{cutoff}"

    # import parameters
    params_full_path = f"s3://fcst-config/forecast-modeling-demand/{environment}.yml"
    params = ut.read_yml(params_full_path)

    print(f"Starting modeling for cutoff {cutoff} in {environment} environment with parameters:")
    ut.pretty_print_dict(params)

    # Define df_jobs
    df_jobs = su.generate_df_jobs(run_name, [cutoff], params['buckets']['refined-data'], params['paths']['run_input_path'])
    df_jobs.apply(lambda row: su.generate_input_data(row, s3fs.S3FileSystem(), params), axis=1)
    # df_jobs = some_function_to_generate_df_jobs()

    # Export parameter on AWS S3

    # Generate data

    # Launch training job

    # Launch batch transform
