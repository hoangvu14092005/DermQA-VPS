# VLMEvalKit + DermNet eval, chạy được trên máy có NVIDIA GPU
# Base image đã có CUDA 12.4 + PyTorch + cuDNN
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # VLMEvalKit tìm dữ liệu (TSV) ở đây
    LMUData=/app/data/LMUData \
    # HuggingFace cache (mount volume để không tải lại model mỗi lần)
    HF_HOME=/app/data/hf_cache

WORKDIR /app

# System deps: git + thư viện cho opencv/cairosvg
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libcairo2 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy source code VLMEvalKit (đã được fix flash-attn / hipho)
COPY VLMEvalKit/ /app/VLMEvalKit/

# Cài dependencies, bỏ polygon3 (cần C++ build tools, không dùng cho DermNet)
RUN grep -v "polygon3" /app/VLMEvalKit/requirements.txt > /app/req.txt && \
    pip install --no-cache-dir -r /app/req.txt && \
    pip install --no-cache-dir -e /app/VLMEvalKit --no-deps && \
    pip install --no-cache-dir "sympy==1.13.1"

# Thư mục dữ liệu (mount khi chạy)
RUN mkdir -p /app/data/LMUData /app/data/images /app/data/hf_cache /app/outputs

# Entry script
COPY docker/run_eval.sh /app/run_eval.sh
RUN chmod +x /app/run_eval.sh

WORKDIR /app/VLMEvalKit
ENTRYPOINT ["/app/run_eval.sh"]
