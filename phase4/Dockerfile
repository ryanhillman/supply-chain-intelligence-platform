FROM python:3.11-slim

WORKDIR /app

# Build dependencies for packages that compile C extensions (numpy, pyarrow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheel first (separate index URL)
# Pinning the CPU build avoids pulling the 2+ GB CUDA variant
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    gcsfs \
    pyarrow \
    fsspec \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    onnx \
    onnxruntime

# Copy project source
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default entrypoint — override with the specific script at runtime:
#   docker run --rm scip-phase4:latest phase4/scripts/train.py --help
ENTRYPOINT ["python"]
