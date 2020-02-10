#!/usr/bin/env bash

# This script builds the Docker image and pushes it to ECR to be ready for use
# The arguments to this script are the image name, the run environment ( dev / prod ) and which weeks to run ( only last = True / False )

algorithm_name=$1
run_env=$2
only_last=$3

# Update docker daemon proxy, this step was necessary for SageMaker notebook instances ( to be corrected by William D. ), but Jenkins is configured right

#sudo cat <<EOF >> /etc/sysconfig/docker
#export HTTPS_PROXY=http://proxy-internet-aws-eu.subsidia.org:3128
#export HTTP_PROXY=http://proxy-internet-aws-eu.subsidia.org:3128
#export NO_PROXY=169.254.169.254,127.0.0.1
#EOF

#sudo service docker restart

# Update shell session proxy
export HTTPS_PROXY=http://proxy-internet-aws-eu.subsidia.org:3128
export HTTP_PROXY=http://proxy-internet-aws-eu.subsidia.org:3128
export NO_PROXY=169.254.169.254,127.0.0.1

# Build image
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
docker build  --no-cache --rm --force-rm --pull --file 'sagemaker_handling/Dockerfile_train' -t ${algorithm_name} \
              --build-arg run_env=${run_env} \
              --build-arg only_last=${only_last} \
              .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
docker rmi ${algorithm_name}
docker rmi ${fullname}