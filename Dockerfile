# fedpod-new training image
# Base: Ubuntu 24.04 + CUDA 12.8 (mirrors fedpod-old v0.4.19-cuda)
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

# ── system packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3.12-dev \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# ── PyTorch 2.7.1 + CUDA 12.8 ───────────────────────────────────────────────
RUN pip3 install --no-cache-dir --break-system-packages \
    torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ── project dependencies ─────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy==1.26.4 \
    pandas==2.2.0 \
    monai==1.3.0 \
    nibabel==5.2.1 \
    natsort==8.4.0

# ── workspace ────────────────────────────────────────────────────────────────
WORKDIR /workspace

# code (data and states are mounted as PVCs at runtime)
COPY run_train.sh         /workspace/run_train.sh
COPY run_aggregation.sh   /workspace/run_aggregation.sh
COPY run_init.sh          /workspace/run_init.sh
COPY scripts/             /workspace/scripts/
COPY experiments/         /workspace/experiments/

RUN chmod +x /workspace/run_train.sh /workspace/run_aggregation.sh /workspace/run_init.sh

# verify
RUN python -c "import torch; print(f'torch {torch.__version__} / cuda {torch.version.cuda}')"

# no ENTRYPOINT — command is set per-step in Argo/Kubeflow
CMD ["python", "-c", "import torch; print(f'fedpod-new | torch {torch.__version__} | cuda {torch.version.cuda}')"]
