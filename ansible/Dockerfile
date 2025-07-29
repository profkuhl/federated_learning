# Use an official PyTorch base image from Docker Hub, which includes CUDA and cuDNN.
# This replaces the nvcr.io image. We choose a recent, stable version.
# ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE_IMAGE}

# The rest of the Dockerfile remains the same.
ARG NVF_BRANCH=main

#RUN apt-get update && \
#    apt-get install -y git && \
#    rm -rf /var/lib/apt/lists/*

# Update pip and install NVFlare. The base image already has python3 and pip.
RUN python3 -m pip install -U pip setuptools
RUN python3 -m pip install nvflare~=2.6.1rc

# Set the working directory and clone the NVFlare examples.
WORKDIR /workspace/
RUN git clone https://github.com/NVIDIA/NVFlare.git --branch ${NVF_BRANCH} --single-branch NVFlare

