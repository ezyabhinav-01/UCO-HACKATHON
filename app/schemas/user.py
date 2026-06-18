"""
app/schemas/user.py

Pydantic schemas for the User resource.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class UserCreate(BaseModel):
    """Payload for creating a new enrolled user."""

    name: str
    email: EmailStr


class UserRead(BaseModel):
    """Basic representation of a user returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: EmailStr
    created_at: datetime


class UserDetailRead(UserRead):
    """
    Extended representation of a user including enrollment status.

    Returned by GET /api/v1/users/{id}.
    """

    is_enrolled: bool
    recording_count: int = 0
    embedding_dimension: int | None = None
    voiceprint_updated_at: datetime | None = None
