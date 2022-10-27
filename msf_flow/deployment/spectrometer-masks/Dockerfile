FROM lambci/lambda:build-python3.7
USER root
ENV AWS_DEFAULT_REGION us-west-2

# copy in requirements
COPY ./requirements.txt .

# install package dependencies
RUN pip install -r requirements.txt --target .
RUN rm -r *.dist-info __pycache__

# copy in source file
COPY ./masks_sds.py ./masks_sds.py

# # create ZIP
# RUN zip -9yr lambda.zip .

# # install packages for running aws login script
# RUN pip install boto requests bs4

# RUN mkdir /.aws

# # RUN echo "aws lambda update-function-code --function-name function-name --zip-file fileb://lambda.zip --profile saml" >> update.sh
# RUN echo "aws s3 cp lambda.zip s3://bucket/deployments/spectrometer-masks.zip --profile saml" >> upload.sh
# RUN chmod u+x upload.sh

ENV AWS True
ENV TXT ''
ENV INPATH data/rdn/ort/
ENV OUTPATH data/masks/

# activate venv and run script
RUN echo "TXT=\$1" >> entrypoint.sh
RUN echo "INPATH=\$2" >> entrypoint.sh
RUN echo "OUTPATH=\$3" >> entrypoint.sh
RUN echo "BUCKET=\$4"
RUN echo "python masks_sds.py --txt \${TXT} --inpath \${INPATH} --outpath \${OUTPATH} --bucket_name \${BUCKET}" >> entrypoint.sh
RUN chmod u+x entrypoint.sh

# ENTRYPOINT [ "python","masks_sds.py" ]
