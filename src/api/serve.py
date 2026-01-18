"""
FastAPI Server for GNN Model Serving
Provides REST API endpoints for threat prediction with monitoring

Author: Mohamed salem eddah
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response

from ..config import get_config
from ..models.gnn_models import create_model
from ..utils.logging import get_logger, setup_logging
from .schemas import (
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)

logger = get_logger(__name__)

# =============================================================================
# Prometheus Metrics
# =============================================================================
REQUEST_COUNT = Counter(
    "gnn_api_requests_total", "Total number of API requests", ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "gnn_api_request_duration_seconds", "Request duration in seconds", ["method", "endpoint"]
)

PREDICTION_COUNT = Counter("gnn_predictions_total", "Total number of predictions made")

INFERENCE_TIME = Histogram(
    "gnn_inference_duration_seconds", "Model inference duration in seconds"
)

ACTIVE_REQUESTS = Gauge("gnn_active_requests", "Number of active requests")

MODEL_LOAD_TIME = Gauge("gnn_model_load_time_seconds", "Time taken to load model")


# =============================================================================
# Global State
# =============================================================================
class ModelServer:
    """Global model server state."""

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.device: str = "cpu"
        self.model_version: str = "unknown"
        self.start_time: float = time.time()
        self.config = None

    def load_model(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        start_time = time.time()
        logger.info(f"Loading model from {checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Create model
            self.model = create_model(
                model_type=self.config.model.type,
                input_dim=checkpoint.get("input_dim", 15),
                hidden_dim=self.config.model.hidden_dim,
                output_dim=2,  # Binary classification
                num_layers=self.config.model.num_layers,
            )

            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self.model_version = checkpoint.get("version", "1.0")

            load_time = time.time() - start_time
            MODEL_LOAD_TIME.set(load_time)

            logger.info(f"Model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    @torch.no_grad()
    def predict(self, features: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """Run inference on input features."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Move to device
            features = features.to(self.device)
            edge_index = edge_index.to(self.device)

            # Inference
            output = self.model(features, edge_index)
            probabilities = torch.softmax(output, dim=1)

            # Get predictions
            predictions = output.argmax(dim=1)

            inference_time = time.time() - start_time
            INFERENCE_TIME.observe(inference_time)

            return {
                "predictions": predictions.cpu().numpy(),
                "probabilities": probabilities.cpu().numpy(),
                "inference_time": inference_time,
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise


# Global server instance
server = ModelServer()


# =============================================================================
# Lifespan Management
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting GNN API Server...")

    # Load configuration
    server.config = get_config()

    # Setup logging
    setup_logging(
        log_level=server.config.logging.level,
        log_dir=server.config.logging.log_dir,
        use_structured_logging=server.config.logging.use_structured_logging,
    )

    # Set device
    server.device = (
        "cuda"
        if server.config.resources.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {server.device}")

    # Load model
    model_path = server.config.output.model_dir / "best_model.pt"
    if model_path.exists():
        try:
            server.load_model(model_path)
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    else:
        logger.warning(f"Model not found at {model_path}")

    yield

    # Shutdown
    logger.info("Shutting down GNN API Server...")


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="GNN Cyber Threat Prediction API",
    description="REST API for network intrusion detection using Graph Neural Networks",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Middleware
# =============================================================================
@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    """Add Prometheus metrics to all requests."""
    ACTIVE_REQUESTS.inc()

    start_time = time.time()

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # Record metrics
        REQUEST_DURATION.labels(method=request.method, endpoint=request.url.path).observe(
            duration
        )
        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.url.path, status=response.status_code
        ).inc()

        return response

    finally:
        ACTIVE_REQUESTS.dec()


# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "GNN Cyber Threat Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - server.start_time

    return HealthResponse(
        status="healthy" if server.model is not None else "degraded",
        model_loaded=server.model is not None,
        uptime_seconds=uptime,
        version=server.model_version,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
async def predict(request: PredictionRequest):
    """
    Predict threat classification for network flows.

    This endpoint accepts network flow features and returns threat predictions.
    """
    if server.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    start_time = time.time()

    try:
        # TODO: Convert flow features to graph representation
        # For now, this is a placeholder that would need proper graph construction

        # Extract features (simplified - would need proper feature engineering)
        num_flows = len(request.flows)

        # Create dummy predictions (replace with actual model inference)
        predictions = []
        for i, flow in enumerate(request.flows):
            # Placeholder logic - replace with actual inference
            prediction = PredictionResult(
                flow_index=i,
                prediction="BENIGN",
                confidence=0.95,
                probabilities={"BENIGN": 0.95, "ATTACK": 0.05},
            )
            predictions.append(prediction)

        PREDICTION_COUNT.inc(num_flows)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return PredictionResponse(
            predictions=predictions,
            model_version=server.model_version,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if server.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    return {
        "model_type": server.config.model.type,
        "version": server.model_version,
        "device": server.device,
        "parameters": sum(p.numel() for p in server.model.parameters()),
    }


# =============================================================================
# Error Handlers
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, timestamp=datetime.utcnow()).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error", detail=str(exc), timestamp=datetime.utcnow()
        ).dict(),
    )


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Run the API server."""
    config = get_config()

    uvicorn.run(
        "src.api.serve:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
