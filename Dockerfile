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
# scipy: required by monai RandRotated (scipy.ndimage)
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy==1.26.4 \
    pandas==2.2.0 \
    monai==1.3.0 \
    nibabel==5.2.1 \
    natsort==8.4.0 \
    scipy>=1.10

# ── workspace ────────────────────────────────────────────────────────────────
WORKDIR /workspace

# Python source code
COPY scripts/             /workspace/scripts/

# Experiment configs: split CSVs + param YAMLs (referenced by relative path in Argo)
COPY experiments/         /workspace/experiments/

# Shell launchers — all entry points called by Argo container args
# Runtime path layout:
#   /workspace/states/  ← checkpoints  (PVC: fedpod-states-pvc)
#   /workspace/logs/    ← training logs (PVC: fedpod-logs-pvc)
#   /data/              ← MRI dataset   (PVC: fets128-data-pvc, read-only)
COPY run_init.sh \
     run_train.sh \
     run_aggregation.sh \
     run_committee.sh \
     run_committee_global.sh \
     run_committee_stage1.sh \
     run_stage1.sh \
     run_stage2_compare.sh \
     run_e2e.sh \
     /workspace/

RUN chmod +x \
    /workspace/run_init.sh \
    /workspace/run_train.sh \
    /workspace/run_aggregation.sh \
    /workspace/run_committee.sh \
    /workspace/run_committee_global.sh \
    /workspace/run_committee_stage1.sh \
    /workspace/run_stage1.sh \
    /workspace/run_stage2_compare.sh \
    /workspace/run_e2e.sh

# Pre-create PVC mount-point directories (populated via volumeMounts at runtime)
RUN mkdir -p /workspace/states /workspace/logs /data

# verify
RUN python -c "import torch; print(f'torch {torch.__version__} / cuda {torch.version.cuda}')"

# no ENTRYPOINT — command is set per-step in Argo/Kubeflow
CMD ["python", "-c", "import torch; print(f'fedpod-new | torch {torch.__version__} | cuda {torch.version.cuda}')"]
