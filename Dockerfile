####################################################################
# DASA-FIDI-IARA - RSNA Pneumonia Detection Challenge Dockerfile
####################################################################
#############################
# General setup
#############################
FROM dasa-fidi-iara/keras-retinanet

# We need git to download some pip packages
RUN apt-get update && apt-get install -y git libsm6

# Upgrade pip
RUN pip install --upgrade pip

# create root directory for our project in the container
RUN mkdir /rsna-challenge-2018

# Set the working directory to /rsna-challenge-2018
WORKDIR /rsna-challenge-2018

# Copy the requirements file into the container at /rsna-challenge-2018/requirements.txt
ADD requirements.txt /rsna-challenge-2018/

# Install required pip packages
# We install Cython in advance as to avoid dependency bugs
RUN pip install Cython==0.28.5 && pip install -r requirements.txt  

# Copy the current directory contents into the container at /rsna-challenge-2018
ADD . /rsna-challenge-2018/

# Create the symbolic link needed by the rsna-challenge module
RUN ln -s /keras-retinanet retinanet/keras_retinanet

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1