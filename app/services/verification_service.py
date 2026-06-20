"""
app/services/verification_service.py

Business logic for the speaker verification + risk evaluation workflow:

    incoming audio -> ECAPA embedding -> fetch enrolled voiceprint
        -> cosine similarity -> identity decision -> risk engine
        -> persist verification_log + risk_log

Used by POST /api/v1/verify.
"""

import uuid

import numpy as np
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.ecapa_service import ECAPAService, get_ecapa_service
from app.repositories.risk_repository import RiskLogRepository
from app.repositories.user_repository import UserRepository
from app.repositories.verification_repository import VerificationLogRepository
from app.repositories.voiceprint_repository import VoiceprintRepository
from app.schemas.verification import VerificationResult
from app.services.risk_engine import compute_risk
from app.utils.audio_utils import cleanup_temp_file, save_upload_to_temp
from app.utils.exceptions import UserNotFoundError, VoiceprintNotFoundError

settings = get_settings()
log = get_logger(__name__)

DECISION_VERIFIED = "verified"
DECISION_MISMATCH = "mismatch"


class VerificationService:
    """Orchestrates the live verification pipeline and risk evaluation."""

    def __init__(
        self,
        session: AsyncSession,
        ecapa_service: ECAPAService | None = None,
    ) -> None:
        self.session = session
        self.user_repo = UserRepository(session)
        self.voiceprint_repo = VoiceprintRepository(session)
        self.verification_log_repo = VerificationLogRepository(session)
        self.risk_log_repo = RiskLogRepository(session)
        self.ecapa_service = ecapa_service or get_ecapa_service()

    async def verify(
        self,
        user_id: uuid.UUID,
        audio_file: UploadFile,
        layer1_score: float = 0.0,
    ) -> VerificationResult:
        """
        Verify a live audio sample against the user's enrolled voiceprint
        and compute the combined Layer1+Layer2 risk assessment.

        Args:
            user_id: The claimed identity (account being accessed).
            audio_file: Live/incoming audio recording.
            layer1_score: AI-voice-detection probability from Layer 1
                           (default 0.0 = "Layer 1 not run / passed cleanly").

        Returns:
            VerificationResult containing the similarity score, identity
            decision, and risk assessment.

        Raises:
            UserNotFoundError: if `user_id` does not correspond to a user.
            VoiceprintNotFoundError: if the user has not completed
                enrollment yet.
        """
        user = await self.user_repo.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError(str(user_id))

        voiceprint = await self.voiceprint_repo.get_by_user_id(user_id)
        if voiceprint is None:
            raise VoiceprintNotFoundError(str(user_id))

        temp_path: str | None = None
        try:
            temp_path = await save_upload_to_temp(audio_file)

            enrolled_embedding = np.asarray(voiceprint.embedding, dtype=np.float32)
            similarity_score = self.ecapa_service.verify_user(
                temp_path, enrolled_embedding
            )

            verified = similarity_score >= settings.SIMILARITY_THRESHOLD
            decision = DECISION_VERIFIED if verified else DECISION_MISMATCH

            risk_result = compute_risk(
                layer1_score=layer1_score, speaker_similarity=similarity_score
            )

            await self.verification_log_repo.create(
                user_id=user_id,
                similarity_score=similarity_score,
                decision=decision,
            )
            await self.risk_log_repo.create(
                user_id=user_id,
                risk_score=risk_result.risk_score,
                risk_level=risk_result.risk_level,
            )
            await self.session.commit()

            log.info(
                f"Verification complete for user_id={user_id}: "
                f"similarity={similarity_score:.4f}, decision={decision}, "
                f"risk_level={risk_result.risk_level}"
            )

            return VerificationResult(
                user_id=user_id,
                similarity_score=similarity_score,
                verified=verified,
                decision=decision,
                layer1_score=layer1_score,
                risk_score=risk_result.risk_score,
                risk_level=risk_result.risk_level,
            )
        except (UserNotFoundError, VoiceprintNotFoundError):
            await self.session.rollback()
            raise
        except Exception:
            await self.session.rollback()
            log.exception(f"Verification failed for user_id={user_id}")
            raise
        finally:
            cleanup_temp_file(temp_path)
