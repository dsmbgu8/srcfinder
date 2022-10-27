FROM lambci/lambda:build-python3.7

USER root

ENV AWS_DEFAULT_REGION us-west-2

# copy in requirements
COPY ./requirements.txt .

# RUN yum -y update
RUN yum -y install sudo wget

# install cmake3
RUN yum -y remove cmake
RUN sudo yum -y install epel-release && \
    sudo yum -y install cmake3 && \
    sudo ln -s /usr/bin/cmake3 /usr/bin/cmake

# create virtual environment
RUN pip install virtualenv
RUN virtualenv python

# install eccodes
RUN wget https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.17.0-Source.tar.gz  && \
    tar xzf eccodes-2.17.0-Source.tar.gz
RUN source python/bin/activate &&  \
    mkdir build ; cd build  && \
    cmake -DCMAKE_INSTALL_PREFIX=/var/task/python/lib/python3.7/site-packages/eccodes-2.17.0 ../eccodes-2.17.0-Source
        # -ENABLE_NETCDF=OFF && \
        # -ENABLE_JPG=OFF && \
        # -ENABLE_PNG=OFF && \
        # -ENABLE_PYTHON=OFF && \
        # -ENABLE_FORTRAN=OFF
RUN cd build && make && \
    sudo make install

RUN source python/bin/activate && pip install eccodes-python numpy==1.18.5 pyproj==2.6.1

# clone pygrib and install locally
RUN git clone https://github.com/jswhit/pygrib 
# copy config in
COPY ./pygrib_setup.cfg ./pygrib/setup.cfg
# RUN cat pygrib/setup.cfg
RUN cd pygrib  && \
    sudo rm -rf .git && \
    zip -r ../pygrib.zip *

# install package dependencies and cleanup extra files
RUN rm -r pygrib
# RUN export PATH=$PATH:/var/task
RUN source python/bin/activate && pip install pygrib.zip
RUN rm pygrib.zip
RUN cd python/lib/python3.7/site-packages && rm -r *.dist-info __pycache__

# create lambda.zip from pip virtualenv packages
# RUN virtualenv deactivate
RUN mkdir package && mv python/ package/python
RUN cd package && zip -r9 ${OLDPWD}/lambda.zip .

# install packages for running aws login script
RUN pip install boto requests bs4

RUN mkdir /.aws

RUN echo "aws s3 cp lambda.zip s3://bucket/deployments/pygrib-layer.zip --profile saml" >> upload.sh
RUN chmod u+x upload.sh
RUN ls
