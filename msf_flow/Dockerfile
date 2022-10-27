FROM lambci/lambda:build-python3.7

USER root

ENV AWS_DEFAULT_REGION us-west-2

# ---------------------------
# Installing pygrib
# ---------------------------
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

# copy in requirements
COPY ./workflow/requirements.txt .

# install package dependencies
RUN source python/bin/activate && pip install eccodes-python numpy==1.18.5 pyproj==2.6.1
RUN source python/bin/activate && pip install -r requirements.txt

# clone pygrib and install locally
RUN git clone https://github.com/jswhit/pygrib 
# copy config in
COPY ./deployment/pygrib/pygrib_setup.cfg ./pygrib/setup.cfg
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
# ---------------------------
# end pygrib install
# ---------------------------

# copy in source files
COPY ./workflow/msf_flow.py .
RUN mkdir ./wind_processor && mkdir ./utils
COPY ./wind_processor/*.py ./wind_processor/
COPY ./utils/dir_watcher.py ./utils/dir_watcher.py
COPY ./utils/logger.py ./utils/logger.py
# COPY ./flist.txt .

# For testing in local docker container
# RUN pip install boto requests bs4
# ENV AWS_PROFILE saml

# env setting for script
ENV AWS TRUE
ENV LD_LIBRARY_PATH /var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/
ENV ECCODES_DEFINITION_PATH /var/task/python/lib/python3.7/site-packages/eccodes-2.17.0/share/eccodes/definitions

# default params to script
ENV PLUMEDIR s3://bucket/data/cmf/ch4/ort/plumes/ch4mfm_v2x1_img_detections/ime_minppmm1500/ang20190926t165202_ch4mf_v2x1_img_ime_minppmm1000.csv
# ENV FILE ang20190926t165202_ch4mf_v2x1_img_ime_minppmm1000.csv
ENV WINDIR s3://bucket/data/wind/
ENV OUTPATH s3://bucket/data/cmf/ch4/ort/plumes/ch4mfm_v2x1_img_detections/plumes_ext

# activate venv and run script
RUN echo "PLUMEDIR=\$1" >> entrypoint.sh
RUN echo "WINDIR=\$2" >> entrypoint.sh
RUN echo "OUTPATH=\$3" >> entrypoint.sh
RUN echo "source python/bin/activate" >> entrypoint.sh
RUN echo "python msf_flow.py -p \${PLUMEDIR} -w \${WINDIR} -o \${OUTPATH}" >> entrypoint.sh
RUN chmod u+x entrypoint.sh
# ["source", "python/bin/activate", "&&", "python msf_flow.py", "-p", "Ref::PLUMEDIR", "-w", "Ref:WINDIR", "-o", "Ref:OUTPATH"]

# ENTRYPOINT ["bash","entrypoint.sh"]
