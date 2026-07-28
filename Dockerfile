# MambaCD - 层级实例级变化检测
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/miniconda/envs/mamba/bin:$PATH"
ENV PYTHONPATH="/workspace/MambaCD:$PYTHONPATH"

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git libgdal-dev gdal-bin python3.11 python3.11-venv python3.11-dev \
    build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda && \
    rm /tmp/miniconda.sh

# Python 环境
RUN /root/miniconda/bin/conda create -n mamba python=3.11 -y

# 安装 PyTorch (CUDA 12.4)
RUN /root/miniconda/envs/mamba/bin/pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装 Python 依赖
COPY requirements.txt /tmp/requirements.txt
RUN /root/miniconda/envs/mamba/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# GDAL Python 绑定
RUN /root/miniconda/envs/mamba/bin/pip install --no-cache-dir \
    GDAL==$(gdal-config --version)

# 编译 CUDA 扩展
COPY kernels/selective_scan /tmp/selective_scan
RUN cd /tmp/selective_scan && \
    /root/miniconda/envs/mamba/bin/pip install --no-cache-dir --no-build-isolation -e . && \
    rm -rf /tmp/selective_scan

# 工作目录
WORKDIR /workspace/MambaCD
COPY . /workspace/MambaCD

# 默认训练命令 (权重和数据通过 volume 挂载)
CMD ["bash", "-c", "source /root/miniconda/bin/activate mamba && \
     python changedetection/script/train_full.py \
     --data_dir /workspace/MambaCD/0617final \
     --scenes 'Airports,Ports,Urban-Rural Areas' \
     --pretrained_weight_path /workspace/MambaCD/weights/vssmtiny_dp01_ckpt_epoch_292.pth \
     --clip_weights_path /workspace/MambaCD/weights/open_clip_pytorch_model.bin \
     --cfg changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml \
     --batch_size 4 --max_epochs 100 --use_amp --num_workers 4"]