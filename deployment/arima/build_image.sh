# Specify an image name
image_name='pmdarima_sagemaker'

# Get account ID
account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration
region=$(aws configure get region)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image_name}:latest"

# If the repository doesn't exist in ECR, create it
aws ecr describe-repositories --repository-names "${image_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --registry-ids ${account} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR with the full name
docker build --no-cache --force-rm --pull --file 'deployment/arima/Dockerfile' -t ${image_name} .
docker tag ${image_name} ${fullname}
docker push ${fullname}

# Some cleaning
docker rmi ${image_name}
docker rmi ${fullname}