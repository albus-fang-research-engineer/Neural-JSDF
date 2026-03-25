# x86_64 PyTorch CUDA base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.23 \
    scipy>=1.10 \
    matplotlib>=3.7 \
    pyyaml>=6.0

    # ===== System dependencies =====
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*
# Fix urdfpy + networkx conflict
RUN pip install --no-cache-dir \
    networkx==2.8.8 \
    rtree \
    open3d \
    lxml\
    trimesh\
    pycollada

RUN pip install --no-cache-dir urdfpy --no-deps
# Set working directory
WORKDIR /root/Neural-JSDF

# Make sure Python can see project modules
ENV PYTHONPATH=/root:$PYTHONPATH
