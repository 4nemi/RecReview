FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    curl \
    && apt-get upgrade -y && apt-get clean

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN useradd -ms /bin/bash ubuntu
#RUN COPY --chown=ubuntu:ubuntu . /home/ubuntu/project/

USER ubuntu
RUN mkdir /home/ubuntu/project

#WORKDIR /home/ubuntu/project