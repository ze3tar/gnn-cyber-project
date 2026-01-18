# Multi-stage Dockerfile for GNN Cyber Project
# Optimized for both development and production use

# =============================================================================
# Base Stage - Common dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =============================================================================
# Dependencies Stage - Install Python packages
# =============================================================================
FROM base as dependencies

# Copy only requirements first for better caching
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# =============================================================================
# Development Stage - For local development
# =============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install -e ".[dev]"

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 9090 5000

# Default command
CMD ["python", "main_pipeline.py", "--help"]

# =============================================================================
# Builder Stage - Build optimized packages
# =============================================================================
FROM dependencies as builder

# Copy application code
COPY . .

# Install the package
RUN pip install --no-deps .

# =============================================================================
# Production Stage - Minimal runtime image
# =============================================================================
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/main_pipeline.py ./
COPY --from=builder /app/config.yaml ./

# Create necessary directories
RUN mkdir -p data/raw data/processed data/graphs data/cache \
    models results logs checkpoints

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Expose ports
EXPOSE 8000 9090

# Default command
CMD ["python", "main_pipeline.py"]

# =============================================================================
# GPU Stage - For GPU-enabled training
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy requirements and install
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed data/graphs data/cache \
    models results logs checkpoints

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('GPU OK')" || exit 1

# Expose ports
EXPOSE 8000 9090

# Default command
CMD ["python", "main_pipeline.py"]

# =============================================================================
# API Server Stage - For model serving
# =============================================================================
FROM production as api

# Install FastAPI and uvicorn if not already installed
RUN pip install fastapi uvicorn[standard] prometheus-client

# Copy API module
COPY src/api ./src/api

# Expose API port
EXPOSE 8000 9090

# Run API server
CMD ["uvicorn", "src.api.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
