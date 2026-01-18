## 🚀 Production Deployment Guide

This guide covers deploying the GNN Cyber Threat Prediction system to production.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [API Server Configuration](#api-server-configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Security Best Practices](#security-best-practices)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum, 32GB+ recommended
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional, for training)
- **Storage**: 100GB+ free space
- **OS**: Ubuntu 20.04+, CentOS 8+, or Docker-compatible OS

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+ (for multi-container setup)
- Python 3.11+ (for local development)
- Git

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/ze3tar/gnn-cyber-project.git
cd gnn-cyber-project
```

### 2. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

**Critical Environment Variables:**

```bash
# Data paths
DATA_RAW_DIR=/path/to/your/CICIDS2017/data
DATA_PROCESSED_DIR=data/processed

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-api-key-here

# Security
SECRET_KEY=your-jwt-secret-key-here

# Monitoring
MLFLOW_TRACKING_URI=http://mlflow:5000
ENABLE_METRICS=true

# Performance
DEVICE=cuda  # or cpu
ENABLE_MIXED_PRECISION=true
BATCH_SIZE=512
```

### 3. Prepare Data

```bash
# Download CICIDS2017 dataset
# Place CSV files in the directory specified by DATA_RAW_DIR

# Verify data directory structure
ls -R $DATA_RAW_DIR
```

---

## Docker Deployment

### Quick Start (Single Container)

```bash
# Build production image
docker build -t gnn-cyber:latest --target production .

# Run container
docker run -d \
  --name gnn-cyber-api \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  gnn-cyber:latest
```

### Full Stack with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

**Available Services:**

- **API Server**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`
- **Prometheus Metrics**: `http://localhost:9090/metrics`
- **Grafana Dashboard**: `http://localhost:3000` (username: admin, password: admin)
- **MLflow UI**: `http://localhost:5000`

### GPU-Enabled Deployment

```bash
# Start GPU-enabled training service
docker-compose --profile gpu up -d gpu-trainer

# Check GPU availability
docker exec gnn-cyber-gpu nvidia-smi
```

---

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace gnn-cyber
```

### 2. Create ConfigMap

```bash
kubectl create configmap gnn-config \
  --from-file=config.yaml \
  --namespace=gnn-cyber
```

### 3. Create Secrets

```bash
kubectl create secret generic gnn-secrets \
  --from-literal=api-key=your-api-key \
  --from-literal=secret-key=your-secret-key \
  --namespace=gnn-cyber
```

### 4. Deploy Application

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gnn-api
  namespace: gnn-cyber
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gnn-api
  template:
    metadata:
      labels:
        app: gnn-api
    spec:
      containers:
      - name: api
        image: gnn-cyber:api
        ports:
        - containerPort: 8000
        - containerPort: 9090
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: gnn-secrets
              key: api-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

```bash
kubectl apply -f k8s/deployment.yaml
```

### 5. Create Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gnn-api-service
  namespace: gnn-cyber
spec:
  type: LoadBalancer
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  selector:
    app: gnn-api
```

```bash
kubectl apply -f k8s/service.yaml
```

---

## API Server Configuration

### Running the API Server

**Option 1: Direct Python**

```bash
python -m src.api.serve
```

**Option 2: Using uvicorn**

```bash
uvicorn src.api.serve:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

**Option 3: Production with Gunicorn**

```bash
gunicorn src.api.serve:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

**Example Request (`sample_request.json`):**

```json
{
  "flows": [
    {
      "source_ip": "192.168.1.100",
      "destination_ip": "10.0.0.50",
      "source_port": 54321,
      "destination_port": 80,
      "protocol": 6,
      "flow_duration": 120000,
      "total_fwd_packets": 10,
      "total_backward_packets": 8,
      "total_length_fwd_packets": 5000,
      "total_length_bwd_packets": 4000,
      "fwd_packet_length_max": 1500,
      "fwd_packet_length_min": 40,
      "fwd_packet_length_mean": 500,
      "fwd_packet_length_std": 200,
      "flow_bytes_s": 75000,
      "flow_packets_s": 150
    }
  ]
}
```

---

## Monitoring Setup

### Prometheus Configuration

Prometheus automatically scrapes metrics from:
- API Server: `http://api:9090/metrics`
- Training Pipeline: `http://app:9090/metrics`

**Key Metrics:**

- `gnn_api_requests_total`: Total API requests
- `gnn_api_request_duration_seconds`: Request latency
- `gnn_predictions_total`: Total predictions made
- `gnn_inference_duration_seconds`: Model inference time
- `gnn_active_requests`: Current active requests

### Grafana Dashboards

1. Access Grafana: `http://localhost:3000`
2. Login with default credentials (admin/admin)
3. Import dashboard from `monitoring/grafana/dashboards/`

**Available Dashboards:**

- API Performance
- Model Metrics
- System Resources

### MLflow Tracking

```bash
# Access MLflow UI
open http://localhost:5000

# Log experiment programmatically
python scripts/train_with_mlflow.py
```

---

## Security Best Practices

### 1. API Key Authentication

```bash
# Generate secure API key
openssl rand -hex 32

# Set in .env
API_KEY=your-generated-key
```

**Usage in requests:**

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/predict
```

### 2. HTTPS/TLS Configuration

```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365

# Run with HTTPS
uvicorn src.api.serve:app \
  --host 0.0.0.0 \
  --port 8443 \
  --ssl-keyfile=key.pem \
  --ssl-certfile=cert.pem
```

### 3. Rate Limiting

Configure in `src/api/serve.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request):
    ...
```

### 4. Input Validation

All inputs are automatically validated using Pydantic schemas in `src/api/schemas.py`.

### 5. Secrets Management

**Option A: Environment Variables**

```bash
export API_KEY=$(cat /run/secrets/api_key)
```

**Option B: HashiCorp Vault**

```bash
vault kv put secret/gnn-cyber api_key=xxx secret_key=yyy
```

**Option C: Kubernetes Secrets**

```bash
kubectl create secret generic gnn-secrets \
  --from-file=api-key=./secrets/api-key.txt
```

---

## Performance Tuning

### 1. Optimize Batch Size

```bash
# Find optimal batch size for your GPU
python scripts/benchmark_batch_size.py
```

### 2. Enable Mixed Precision

```bash
export ENABLE_MIXED_PRECISION=true
```

### 3. Adjust Worker Count

```bash
# CPU workers = 2 * num_cores + 1
export API_WORKERS=9  # for 4-core system
```

### 4. Model Quantization

```python
from src.utils.mlops import ModelOptimizer

# Quantize model for faster inference
quantized_model = ModelOptimizer.quantize_dynamic(model)
```

### 5. Caching Strategy

```bash
# Enable caching
export ENABLE_CACHING=true
export CACHE_FORMAT=parquet  # 10x faster than CSV
```

---

## Troubleshooting

### Common Issues

**Issue 1: Out of Memory (OOM)**

```bash
# Reduce batch size
export BATCH_SIZE=256

# Enable gradient accumulation
export GRADIENT_ACCUMULATION_STEPS=4
```

**Issue 2: Slow Data Loading**

```bash
# Convert CSV to Parquet
python scripts/convert_to_parquet.py

# Increase num_workers
export NUM_WORKERS=8
```

**Issue 3: API Timeouts**

```bash
# Increase timeout
uvicorn src.api.serve:app --timeout-keep-alive 300
```

**Issue 4: Model Not Found**

```bash
# Check model path
ls -la models/

# Verify config
cat config.yaml | grep model_dir
```

### Logging

```bash
# View API logs
docker-compose logs -f api

# View training logs
tail -f logs/pipeline_*.log

# Enable debug logging
export LOG_LEVEL=DEBUG
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Metrics endpoint
curl http://localhost:9090/metrics

# Docker container health
docker ps --filter health=healthy
```

---

## Performance Benchmarks

### Expected Performance

| Metric | Value |
|--------|-------|
| Inference Latency (CPU) | ~20-50ms per flow |
| Inference Latency (GPU) | ~5-15ms per flow |
| Throughput (CPU) | ~500-1000 flows/sec |
| Throughput (GPU) | ~2000-5000 flows/sec |
| Model Size | ~10-50MB |
| Memory Usage | ~2-4GB |

### Optimization Results

| Optimization | Speed Improvement | Memory Reduction |
|--------------|-------------------|------------------|
| Mini-batch sampling | 3-5x | 60% |
| Mixed precision | 1.4x | 40% |
| Model quantization | 2-4x | 75% |
| Parquet caching | 10x loading | - |
| ONNX export | 1.5-2x | - |

---

## Maintenance

### Backup Strategy

```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup database (if using)
pg_dump gnn_cyber > backup.sql
```

### Model Updates

```bash
# Train new model
python main_pipeline.py --mode train

# Test new model
python main_pipeline.py --mode evaluate

# Deploy new model (zero-downtime)
kubectl rollout restart deployment/gnn-api
```

### Log Rotation

```bash
# Configure log rotation
cat > /etc/logrotate.d/gnn-cyber <<EOF
/app/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 appuser appuser
}
EOF
```

---

## Support

For issues and questions:

- GitHub Issues: https://github.com/ze3tar/gnn-cyber-project/issues
- Documentation: https://github.com/ze3tar/gnn-cyber-project/wiki
- Email: your.email@example.com
