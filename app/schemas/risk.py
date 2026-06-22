"""
app/schemas/risk.py

Pydantic schemas for the Risk Engine.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RiskInput(BaseModel):
    """
    Input to the Risk Engine.

    layer1_score: AI-voice-detection probability from Layer 1 (0.0 - 1.0).
    speaker_similarity: Cosine similarity from Layer 2 ECAPA-TDNN verification
                         (-1.0 - 1.0, practically 0.0 - 1.0).
    """

    layer1_score: float = Field(ge=0.0, le=1.0)
    speaker_similarity: float = Field(ge=-1.0, le=1.0)


class RiskResult(BaseModel):
    """Output of the Risk Engine."""

    risk_score: float
    risk_level: str  # "CLEAN" | "FRAUD_ALERT"


class RiskLogRead(BaseModel):
    """A single historical risk evaluation record."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    risk_score: float
    risk_level: str
    created_at: datetime
