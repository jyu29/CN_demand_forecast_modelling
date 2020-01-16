#!/usr/bin/env bash

# This script builds the Docker image and pushes it to ECR to be ready for use
# The argument to this script is the image name.

image=$1
algorithm_name=apo-demand-forecast-training #$1

# Update docker daemon proxy
sudo su

cat <<EOF >> /etc/sysconfig/docker
export HTTPS_PROXY=http://proxy-internet-aws-eu.subsidia.org:3128
export HTTP_PROXY=http://proxy-internet-aws-eu.subsidia.org:3128
export NO_PROXY=169.254.169.254,127.0.0.1
EOF

service docker restart
exit

# Update shell session proxy
export HTTPS_PROXY=http://proxy-internet-aws-eu.subsidia.org:3128
export HTTP_PROXY=http://proxy-internet-aws-eu.subsidia.org:3128
export NO_PROXY=169.254.169.254,127.0.0.1

# Build image
cd container
chmod -R +x src/

# Get FCST United ECR account ID
account=$(aws sts get-caller-identity --query Account --output text)
# Get the region defined in the current configuration
region=$(aws configure get region)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist in ECR, create it
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --registry-ids ${account} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build  --no-cache --pull --file 'Dockerfile_train' -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}