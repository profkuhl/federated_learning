# Use the latest NVIDIA PyTorch container (includes CUDA, cuDNN, PyTorch optimized for NVIDIA GPUs)
# Format: YY.MM-py3 (e.g., 25.10 = October 2025)
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

# Update pip and install NVFlare 2.7.0 (latest stable release)
RUN python3 -m pip install -U pip setuptools wheel && \
    python3 -m pip install nvflare==2.7.0 && \
    python3 -m pip install torch torchvision numpy pandas polars

# Set working directory
WORKDIR /workspace/

# Verify CUDA is available (optional - shows CUDA version in build logs)
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

