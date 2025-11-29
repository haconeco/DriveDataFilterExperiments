# SegFormer inference environment for RoadLib
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# OS dependencies (include RoadLib C++ prerequisites)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git wget curl ca-certificates \
    build-essential ffmpeg libgl1 libglib2.0-0 \
    libopencv-dev libpcl-dev libglfw3-dev libceres-dev libeigen3-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Python tooling
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch (CUDA 11.8 wheels)
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Optional dataset/utility dependencies (Install numpy<2.0 first to avoid conflicts)
RUN python3 -m pip install --no-cache-dir \
    "numpy<2.0" opencv-python pycocotools pillow scipy tqdm rich matplotlib nuscenes-devkit ftfy regex

# OpenMMLab core dependencies
RUN python3 -m pip install --no-cache-dir openmim==0.3.9 && \
    mim install --yes "mmengine==0.10.4" "mmcv==2.1.0" "mmsegmentation==1.2.2"



WORKDIR /workspace

# Clone MMSegmentation source for configs/tools (kept in editable mode)
RUN git clone --depth 1 --branch v1.2.2 https://github.com/open-mmlab/mmsegmentation.git /workspace/mmsegmentation && \
    python3 -m pip install --no-cache-dir -e /workspace/mmsegmentation

# Clone RoadLib (SegFormer config/inference reference)
RUN git clone --depth 1 https://github.com/GREAT-WHU/RoadLib.git /workspace/RoadLib

# Fetch SegFormer config into MMSegmentation config tree
RUN mkdir -p /workspace/mmsegmentation/configs/segformer && \
    curl -L https://raw.githubusercontent.com/GREAT-WHU/RoadLib/master/scripts/segformer_whu.py \
    -o /workspace/mmsegmentation/configs/segformer/segformer_whu.py

# Runtime directories for data and checkpoints
RUN mkdir -p /workspace/data/nuscenes /workspace/weights /workspace/output

ENV PYTHONPATH=/workspace/mmsegmentation:/workspace/RoadLib:${PYTHONPATH} \
    TORCH_HOME=/workspace/.cache/torch

CMD ["/bin/bash"]
