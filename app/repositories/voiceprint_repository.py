"""
app/repositories/voiceprint_repository.py

Data-access layer for the `voiceprints` table.

Voiceprints store the averaged ECAPA-TDNN embedding for a user as a
pgvector VECTOR column. This repository handles both creation (first-time
enrollment) and updates (re-enrollment, which overwrites the existing
voiceprint).
"""

import uuid

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.voiceprint import Voiceprint


class VoiceprintRepository:
    """Encapsulates all database operations for the Voiceprint model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_user_id(self, user_id: uuid.UUID) -> Voiceprint | None:
        """Fetch the voiceprint belonging to a given user, if any."""
        result = await self.session.execute(
            select(Voiceprint).where(Voiceprint.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        user_id: uuid.UUID,
        embedding: np.ndarray,
        recording_count: int,
    ) -> Voiceprint:
        """
        Create a new voiceprint for `user_id`, or overwrite the existing one
        if enrollment is being re-run.

        Args:
            user_id: The owning user's id.
            embedding: 1-D numpy array (the averaged ECAPA-TDNN embedding).
            recording_count: Number of utterances used to build the
                              embedding.

        Returns:
            The persisted Voiceprint instance.
        """
        embedding_list = embedding.astype(float).tolist()

        existing = await self.get_by_user_id(user_id)
        if existing is not None:
            existing.embedding = embedding_list
            existing.recording_count = recording_count
            self.session.add(existing)
            await self.session.flush()
            await self.session.refresh(existing)
            return existing

        voiceprint = Voiceprint(
            user_id=user_id,
            embedding=embedding_list,
            recording_count=recording_count,
        )
        self.session.add(voiceprint)
        await self.session.flush()
        await self.session.refresh(voiceprint)
        return voiceprint

    async def delete_by_user_id(self, user_id: uuid.UUID) -> bool:
        """Delete the voiceprint for a user, if one exists. Returns True if deleted."""
        existing = await self.get_by_user_id(user_id)
        if existing is None:
            return False
        await self.session.delete(existing)
        await self.session.flush()
        return True
