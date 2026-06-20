"""
app/repositories/verification_repository.py

Data-access layer for the `verification_logs` table.
"""

import uuid

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.verification_log import VerificationLog


class VerificationLogRepository:
    """Encapsulates all database operations for the VerificationLog model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self, user_id: uuid.UUID, similarity_score: float, decision: str
    ) -> VerificationLog:
        """Insert a new verification log entry."""
        log_entry = VerificationLog(
            user_id=user_id,
            similarity_score=similarity_score,
            decision=decision,
        )
        self.session.add(log_entry)
        await self.session.flush()
        await self.session.refresh(log_entry)
        return log_entry

    async def list_by_user_id(
        self, user_id: uuid.UUID, limit: int = 50, offset: int = 0
    ) -> list[VerificationLog]:
        """
        Return verification history for a user, most recent first.

        Args:
            user_id: The user to fetch history for.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).
        """
        result = await self.session.execute(
            select(VerificationLog)
            .where(VerificationLog.user_id == user_id)
            .order_by(desc(VerificationLog.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
