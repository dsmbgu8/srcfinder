#!/usr/bin/env bash

CODE_PATH=$1
IMAGE_NAME=$2 # Image name must match the name on AWS


# Our ECR repository and region
#Update the REPO_NAME value
REPO_NAME=$3
AWS_REGION="us-west-2"


echo "==> build docker image"
(cd ${CODE_PATH} && docker build -t ${IMAGE_NAME} .)

echo "==> get aws credentials and login to repo"
# Make sure you have logged into aws before running the below command and upde ecr to your saml-pub repo
$(aws --profile saml-pub ecr get-login --no-include-email --region ${AWS_REGION})




echo "==> checking if repo exists"
if aws --profile saml-pub ecr describe-repositories --repository-names $IMAGE_NAME ; then
    echo "==> repo exists"
else
    # If function does not exist, create function - modify to specify runtime, role, etc.
    echo "==> repo does not exist - creating repo"
    aws ecr --profile saml-pub create-repository --repository-name $IMAGE_NAME
fi

echo "==> tag and push image"
(cd ${CODE_PATH} && docker tag ${IMAGE_NAME}:latest ${REPO_NAME}/${IMAGE_NAME}:latest)
(cd ${CODE_PATH} && docker push ${REPO_NAME}/${IMAGE_NAME}:latest)
