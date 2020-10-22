"""
Python script to orchestrate demand forecast modeling:
- Engineers features & prepares data in DeepAR format
- Creates Training Docker Image
- Pops instance to  preprocess train a DeepAR model, then output predictions
@author: benbouillet ( Benjamin Bouillet )
"""
import src.utils as ut
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', choices=['dev', 'prod'], default="dev",
                        help="'dev' or 'prod', to set the right configurations")
    parser.add_argument('--cutoff', default=ut.get_current_week(), help="cutoff in format YYYYWW or 'today'")
    args = parser.parse_args()
    environment = args.environment
    cutoff = args.cutoff

    # import parameters
    params_full_path = f"s3://fcst-config/forecast-modeling-demand/{environment}.json"
    params = ut.read_json(params_full_path)

    print(f"Starting modeling for cutoff {cutoff} in {environment} environment with parameters:")
    ut.pretty_print_json(params)

    # Define df_jobs
    # df_jobs = some_function_to_generate_df_jobs()

    # Export parameter on AWS S3

    # Generate data

    # Launch training job

    # Launch batch transform
