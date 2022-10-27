FROM lambci/lambda:build-python3.7

USER root

ENV AWS_DEFAULT_REGION us-west-2

# copy in requirements
COPY ./requirements.txt .

# install package dependencies
RUN pip install -r requirements.txt --target .
RUN rm -r *.dist-info __pycache__

# copy in source files
COPY ./cluster_incr_nn.py .
 
# create ZIP
RUN zip -9yr lambda.zip .

# install packages for running aws login script
RUN pip install boto3 requests bs4 pytz lxml


# env setting for script
# ENV AWS TRUE

# default params to script
# ENV PLUMELIST data/plumes/ang20190926t165202_ch4mf_v2x1_img_ime_minppmm1000.csv
# ENV OUTPATH data/products/cluster_out.csv
# ENV RADIUS 150

# RUN echo "python cluster_incr_nn.py $PLUMELIST $OUTPATH -r $RADIUS" >> entrypoint.sh
# RUN chmod u+x entrypoint.sh

# ENTRYPOINT ["bash","entrypoint.sh"]

# RUN echo "#!/bin/bash \naws lambda update-function-code --function-name plume-clustering --zip-file fileb://lambda.zip --profile saml" >> update.sh
RUN echo "aws s3 cp lambda.zip s3://bucket/deployments/plume-clustering.zip --profile saml-pub" >> upload.sh
RUN chmod u+x upload.sh
