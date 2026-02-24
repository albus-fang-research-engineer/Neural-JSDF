FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU, ARM64 wheel)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Your deps
RUN pip install --no-cache-dir \
    numpy>=1.23 \
    scipy>=1.10 \
    matplotlib>=3.7 \
    pyyaml>=6.0

WORKDIR /root/Neural-JSDF
ENV PYTHONPATH=/root:$PYTHONPATH