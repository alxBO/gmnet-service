"""Pydantic models for API request/response schemas."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    width: int
    height: int
    file_size_bytes: int
    format: str
    histogram: Dict[str, List[int]]
    dynamic_range_ev: float
    mean_brightness: float
    median_brightness: float
    mean_luminance_linear: float
    peak_luminance_linear: float
    contrast_ratio: float
    min_luminance_linear: float = 0.0
    clipping_percent: float


class GenerateRequest(BaseModel):
    model: Literal["synthetic", "realworld"] = "realworld"
    scale: Literal[1, 2, 4] = 1
    peak: float = Field(default=8.0, ge=2.0, le=32.0)


class ProgressEvent(BaseModel):
    stage: str
    progress: float
    message: str
    queue_position: int = 0


class HdrAnalysis(BaseModel):
    dynamic_range_ev: float
    contrast_ratio: float
    min_luminance: float = 0.0
    peak_luminance: float
    mean_luminance: float
    luminance_percentiles: Dict[str, float]
    hdr_histogram: dict


class ResultResponse(BaseModel):
    job_id: str
    download_url: str
    analysis: HdrAnalysis
    processing_time_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
