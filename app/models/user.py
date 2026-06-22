"""
app/models/user.py

SQLAlchemy ORM model for the `users` table.

Represents an enrolled customer / bank account holder whose voice will be
verified against incoming calls.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class User(Base):
    """A bank customer enrolled in PhaseGuard's voice verification system."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    voiceprint: Mapped["Voiceprint"] = relationship(  # noqa: F821
        "Voiceprint",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    verification_logs: Mapped[list["VerificationLog"]] = relationship(  # noqa: F821
        "VerificationLog",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    risk_logs: Mapped[list["RiskLog"]] = relationship(  # noqa: F821
        "RiskLog",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} name={self.name!r} email={self.email!r}>"
