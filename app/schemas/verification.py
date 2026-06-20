"""
app/schemas/verification.py

Pydantic schemas for speaker verification requests, results, and history.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class VerificationRequestMeta(BaseModel):
    """
    Optional metadata accompanying a verification request.

    `layer1_score` represents the AI-voice-detection probability produced by
    Layer 1 (0.0 = certainly human, 1.0 = certainly AI-generated). If Layer 1
    has not been run yet (e.g. Layer 2 is being tested in isolation), this
    defaults to 0.0, meaning "Layer 1 passed cleanly".
    """

    layer1_score: float = Field(default=0.0, ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """Response payload returned by POST /api/v1/verify."""

    user_id: uuid.UUID
    similarity_score: float
    verified: bool
    decision: str  # "verified" | "mismatch"

    # Risk engine output (Layer1 + Layer2 combined)
    layer1_score: float
    risk_score: float
    risk_level: str  # "CLEAN" | "FRAUD_ALERT"


class VerificationLogRead(BaseModel):
    """A single historical verification record."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    similarity_score: float
    decision: str
    created_at: datetime
