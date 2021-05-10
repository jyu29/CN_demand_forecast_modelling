FROM 150258775384.dkr.ecr.eu-west-1.amazonaws.com/python:3.7-slim-buster

# Set proxy variables
ENV http_proxy=http://proxy-internet-aws-eu.subsidia.org:3128
ENV https_proxy=http://proxy-internet-aws-eu.subsidia.org:3128
ENV no_proxy=169.254.169.254,127.0.0.1,*.s3-eu-west-1.amazonaws.com,*.s3.eu-west-1.amazonaws.com,s3.eu-west-1.amazonaws.com

# Copies all needed files inside the container
COPY src/ deployment/arima/requirements.txt ./

# Install requirements
RUN pip3 install -r requirements.txt

# Defines the main script as script entry point
ENTRYPOINT ["python3", "main_sagemaker_arima.py"]
