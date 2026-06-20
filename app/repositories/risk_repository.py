"""
app/repositories/risk_repository.py

Data-access layer for the `risk_logs` table.
"""

import uuid

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.risk_log import RiskLog


class RiskLogRepository:
    """Encapsulates all database operations for the RiskLog model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self, user_id: uuid.UUID, risk_score: float, risk_level: str
    ) -> RiskLog:
        """Insert a new risk log entry."""
        log_entry = RiskLog(
            user_id=user_id,
            risk_score=risk_score,
            risk_level=risk_level,
        )
        self.session.add(log_entry)
        await self.session.flush()
        await self.session.refresh(log_entry)
        return log_entry

    async def list_by_user_id(
        self, user_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> list[RiskLog]:
        """Return risk evaluation history for a user, most recent first."""
        result = await self.session.execute(
            select(RiskLog)
            .where(RiskLog.user_id == user_id)
            .order_by(desc(RiskLog.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
