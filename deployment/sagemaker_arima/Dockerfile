FROM 150258775384.dkr.ecr.eu-west-1.amazonaws.com/python:3.7-slim-buster

# Set proxy variables
ENV http_proxy=http://proxy-internet-aws-eu.subsidia.org:3128
ENV https_proxy=http://proxy-internet-aws-eu.subsidia.org:3128
ENV no_proxy=127.0.0.1,localhost,169.254.169.254,.emr.preprod.subsidia.org,.subsidia.org,.preprod.subsidia.org,.hosting.eu,.preprod.org,.aws.eu-west-1.hosting.eu,s3-eu-west-1.amazonaws.com,*.s3-eu-west-1.amazonaws.com,*.s3.eu-west-1.amazonaws.com,s3.eu-west-1.amazonaws.com

# Set useful paths in environment variables
ENV SOURCE_CODE_DIR /opt/ml/code/
ENV INPUT_DATA_DIR /opt/ml/input/data/train/
ENV HPS_FILE_PATH /opt/ml/input/config/hyperparameters.json

# Copy all needed files inside the container
COPY src/ deployment/sagemaker_arima/requirements.txt ${SOURCE_CODE_DIR}

# Set defaut workdir for docker
WORKDIR ${SOURCE_CODE_DIR}

# Install requirements
RUN pip3 install -r requirements.txt

# Defines the main script as script entry point
ENTRYPOINT ["python3", "main_sagemaker_arima.py"]