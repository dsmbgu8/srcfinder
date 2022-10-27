FROM lambci/lambda:build-python3.7

USER root

ENV AWS_DEFAULT_REGION us-west-2

# copy in requirements
COPY ./requirements.txt .

# install package dependencies
RUN pip install -r requirements.txt --target .
RUN rm -r *.dist-info __pycache__

# copy in source file
COPY ./source_persistence.py .

# create ZIP
RUN zip -9yr lambda.zip .

# install packages for running aws login script
RUN pip install boto requests bs4

RUN mkdir /.aws

# RUN echo "#!/bin/bash \naws lambda update-function-code --function-name source-persistence --zip-file fileb://lambda.zip --profile saml" >> update.sh
RUN echo "aws s3 cp lambda.zip s3://bucket/deployments/source-persistence.zip --profile saml" >> upload.sh
RUN chmod u+x upload.sh
