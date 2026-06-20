"""
app/schemas/enrollment.py

Pydantic schemas for the voice enrollment workflow.
"""

import uuid

from pydantic import BaseModel


class EnrollmentResponse(BaseModel):
    """Response payload returned by POST /api/v1/enroll."""

    success: bool
    user_id: uuid.UUID
    recording_count: int
    embedding_dimension: int
    message: str
