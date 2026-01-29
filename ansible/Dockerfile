# Use NVIDIA PyTorch container compatible with driver 570.x
# Format: YY.MM-py3 (e.g., 24.12 = December 2024)
# Note: 25.10+ requires driver 580.95+, but we have driver 570.195.03
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}

# Update pip and install NVFlare 2.7.0 (latest stable release)
RUN python3 -m pip install -U pip setuptools wheel && \
    python3 -m pip install nvflare==2.7.1 && \
    python3 -m pip install torch torchvision numpy pandas polars

# Set working directory
WORKDIR /workspace/

# Verify CUDA is available (optional - shows CUDA version in build logs)
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

