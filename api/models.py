"""
Task 7 — FastAPI Inference System [30 Marks]

Pydantic models for request/response validation.
- Text: 10–10,000 chars
- top_k: 1–20
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from enum import Enum


# ──────────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────────
class ProcessingStep(str, Enum):
    CLEAN = "clean"
    TOKENIZE = "tokenize"
    REMOVE_STOPWORDS = "remove_stopwords"
    NORMALIZE = "normalize"


# ──────────────────────────────────────────────────────────
#  Request Models
# ──────────────────────────────────────────────────────────
class PreprocessRequest(BaseModel):
    """Request model for /preprocess endpoint."""
    text: str = Field(..., min_length=10, max_length=10000,
                      description="Text to preprocess (10-10,000 chars)")
    steps: List[ProcessingStep] = Field(
        default=[ProcessingStep.CLEAN, ProcessingStep.TOKENIZE, 
                 ProcessingStep.REMOVE_STOPWORDS, ProcessingStep.NORMALIZE],
        description="Processing steps to apply"
    )


class ClassifyRequest(BaseModel):
    """Request model for /classify endpoint."""
    text: str = Field(..., min_length=10, max_length=10000,
                      description="Text to classify (10-10,000 chars)")


class BatchClassifyRequest(BaseModel):
    """Request model for /classify/batch endpoint."""
    texts: List[str] = Field(..., min_length=1, max_length=100,
                              description="List of texts to classify (1-100 texts)")
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 texts per batch")
        for i, text in enumerate(v):
            if len(text) < 10:
                raise ValueError(f"Text at index {i} is too short (min 10 chars)")
            if len(text) > 10000:
                raise ValueError(f"Text at index {i} is too long (max 10,000 chars)")
        return v


class SimilarRequest(BaseModel):
    """Request model for /retrieve/similar endpoint."""
    text: str = Field(..., min_length=10, max_length=10000,
                      description="Query text (10-10,000 chars)")
    top_k: int = Field(default=5, ge=1, le=20,
                       description="Number of similar claims to return (1-20)")


# ──────────────────────────────────────────────────────────
#  Response Models
# ──────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    model_name: str
    model_version: str
    model_stage: str
    f1_score: float
    load_timestamp: str
    status: str = "healthy"


class PreprocessResponse(BaseModel):
    """Response model for /preprocess endpoint."""
    original_text: str
    tokens: List[str]
    removed_stopwords: List[str]
    processing_steps: List[str]
    processing_time_ms: float
    token_count: int


class ClassifyResponse(BaseModel):
    """Response model for /classify endpoint."""
    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]
    top_features: List[Dict[str, float]]
    processing_time_ms: float


class BatchClassifyResponse(BaseModel):
    """Response model for /classify/batch endpoint."""
    results: List[ClassifyResponse]
    total_texts: int
    total_processing_time_ms: float


class SimilarClaim(BaseModel):
    """A single similar claim result."""
    text: str
    label: str
    similarity_score: float
    rank: int


class SimilarResponse(BaseModel):
    """Response model for /retrieve/similar endpoint."""
    query: str
    similar_claims: List[SimilarClaim]
    processing_time_ms: float


class PerformanceMetrics(BaseModel):
    """Response model for /model/performance endpoint."""
    current_model: str
    current_version: str
    metrics: Dict[str, float]
    version_history: List[Dict[str, Optional[str]]]
    last_updated: str
