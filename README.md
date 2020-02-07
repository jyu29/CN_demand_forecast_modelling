# Demand Forecast : Modeling brick

## Context

This project represents the ML brick of the Demand Forecasting project for APO.

The run pipeline for the entire project is represented hereunder:
![Pipeline model](assets/pipeline_modelisation.png)

Here are the GitHub links to the other blocks of the project:
- [Data Ingestion](https://github.com/dktunited/forecast-data-ingestion.git)
- [Data Refining](https://github.com/dktunited/forecast-data-refining-demand.git)
- [Monitoring](https://github.com/dktunited/forecast-monitoring.git)


## Description

This ML block is orchestrated by a `main.py` file with two command options:
- `--environment`: which can either be `dev` or `prod` ( sets the configuration file to use for the project input and output data )
- `--only_last`: which can either be `True` or `False` ( determines if the ML brick should run for the last cutoff only or for all available cutoffs )

This `main.py` file does the following:
- First, it runs `_sagemaker_/build_image.sh` script to build the docker training image for SageMaker
- Then, it creates a **SageMaker Training Job** through a `boto3` API call
- Lastly, it monitors the status of the training job through `boto3` API calls as well and reports on it status.

## Scheduling

This brick is scheduled through Jenkins:
- [Jenkins job](https://forecast-jenkins.subsidia.org/view/PIPELINE-RUN/job/forecast-modeling-demand/)
- The hereinabove job is called upon by this [Run Pipeline](https://forecast-jenkins.subsidia.org/job/forecast-pipeline-demand/) job


