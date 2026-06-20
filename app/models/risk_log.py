"""
app/models/risk_log.py

SQLAlchemy ORM model for the `risk_logs` table.

Each row records the output of the Risk Engine for a given verification
event: a combined risk score and categorical risk level.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class RiskLog(Base):
    """A historical record of one Risk Engine evaluation."""

    __tablename__ = "risk_logs"

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
        index=True,
    )
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)

    # "CLEAN" | "FRAUD_ALERT"
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="risk_logs")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<RiskLog id={self.id} user_id={self.user_id} "
            f"risk_score={self.risk_score:.2f} risk_level={self.risk_level!r}>"
        )
