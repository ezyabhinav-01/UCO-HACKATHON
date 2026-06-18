"""
app/schemas/voiceprint.py

Pydantic schemas for the Voiceprint resource.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class VoiceprintRead(BaseModel):
    """Representation of a stored voiceprint (embedding metadata only)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    recording_count: int
    created_at: datetime
    updated_at: datetime
