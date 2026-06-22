"""
app/models/voiceprint.py

SQLAlchemy ORM model for the `voiceprints` table.

Stores the averaged ECAPA-TDNN speaker embedding (192-dimensional vector)
for an enrolled user, persisted via pgvector's VECTOR column type so that
similarity search can be performed directly inside PostgreSQL if desired.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.database.base import Base

settings = get_settings()


class Voiceprint(Base):
    """
    A single enrolled voiceprint (averaged speaker embedding) for a user.

    Each user has exactly one active voiceprint, created/updated by the
    enrollment workflow (averaging embeddings from 30-50 utterances).
    """

    __tablename__ = "voiceprints"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # pgvector column storing the 192-dimensional ECAPA-TDNN embedding.
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.EMBEDDING_DIM), nullable=False
    )

    recording_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="voiceprint")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<Voiceprint id={self.id} user_id={self.user_id} "
            f"recording_count={self.recording_count}>"
        )
