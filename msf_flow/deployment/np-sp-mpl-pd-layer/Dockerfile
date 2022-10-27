FROM lambci/lambda:build-python3.7

USER root
ENV AWS_DEFUALT_REGION us-west-2

COPY ./requirements.txt .

# create virtual environment
RUN virtualenv python

# activate virtualenv and install packages
RUN source python/bin/activate && \
    pip install -r requirements.txt

# cleanup extra files
RUN cd python/lib/python3.7/site-packages && \
    rm -r *.dist-info __pycache__

# create lambda.zip from pip virtualenv packages
RUN mkdir package && mv python/ package/python
RUN cd package && zip -r9 ${OLDPWD}/layer.zip *

# install packages for running aws login script
RUN pip install boto requests bs4

# create directory for aws creds to be written to
RUN mkdir ~/.aws

RUN echo "aws s3 cp layer.zip s3://bucket/deployments/np-sp-mpl-pd-layer.zip --profile saml" >> upload.sh
RUN chmod u+x upload.sh

# RUN echo "./upload.sh" >> entrypoint.sh
# RUN chmod u+x entrypoint.sh

# ENTRYPOINT [ "bash", "entrypoint.sh" ]
