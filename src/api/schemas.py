"""
Pydantic schemas for API request/response validation
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class FlowFeatures(BaseModel):
    """Network flow features for prediction."""

    source_ip: str = Field(..., description="Source IP address")
    destination_ip: str = Field(..., description="Destination IP address")
    source_port: int = Field(..., ge=0, le=65535, description="Source port")
    destination_port: int = Field(..., ge=0, le=65535, description="Destination port")
    protocol: int = Field(..., description="Protocol number (6=TCP, 17=UDP)")
    flow_duration: float = Field(..., ge=0, description="Flow duration in microseconds")
    total_fwd_packets: int = Field(..., ge=0, description="Total forward packets")
    total_backward_packets: int = Field(..., ge=0, description="Total backward packets")
    total_length_fwd_packets: float = Field(..., ge=0)
    total_length_bwd_packets: float = Field(..., ge=0)
    fwd_packet_length_max: float = Field(..., ge=0)
    fwd_packet_length_min: float = Field(..., ge=0)
    fwd_packet_length_mean: float = Field(..., ge=0)
    fwd_packet_length_std: float = Field(..., ge=0)
    flow_bytes_s: float = Field(..., ge=0)
    flow_packets_s: float = Field(..., ge=0)

    class Config:
        schema_extra = {
            "example": {
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
                "flow_packets_s": 150,
            }
        }


class PredictionRequest(BaseModel):
    """Request for threat prediction."""

    flows: List[FlowFeatures] = Field(..., min_items=1, max_items=1000)

    @validator("flows")
    def validate_flows(cls, v):
        if len(v) > 1000:
            raise ValueError("Maximum 1000 flows per request")
        return v


class PredictionResult(BaseModel):
    """Individual prediction result."""

    flow_index: int
    prediction: str = Field(..., description="Prediction label (BENIGN or ATTACK)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")


class PredictionResponse(BaseModel):
    """Response containing predictions."""

    predictions: List[PredictionResult]
    model_version: str
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str


class MetricsResponse(BaseModel):
    """Model performance metrics."""

    total_predictions: int
    average_inference_time_ms: float
    requests_per_second: float
    model_info: Dict[str, any]


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
