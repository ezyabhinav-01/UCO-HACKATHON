"""
tests/test_verification_service.py

Unit tests for VerificationService that mock out the database session,
ECAPA service, and file utilities.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.schemas.verification import VerificationResult
from app.services.verification_service import (
    DECISION_MISMATCH,
    DECISION_VERIFIED,
    VerificationService,
)
from app.utils.exceptions import UserNotFoundError, VoiceprintNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_user(user_id: uuid.UUID | None = None) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid.uuid4()
    user.name = "Test User"
    return user


def _make_fake_voiceprint(user_id: uuid.UUID, embedding_dim: int = 192) -> MagicMock:
    vp = MagicMock()
    vp.id = uuid.uuid4()
    vp.user_id = user_id
    vp.embedding = np.random.rand(embedding_dim).tolist()
    return vp


def _make_upload_file(name: str = "live.wav") -> MagicMock:
    uf = MagicMock()
    uf.filename = name
    uf.content_type = "audio/wav"
    return uf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_raises_user_not_found():
    session = AsyncMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = None

    with patch.object(service, "user_repo", user_repo_mock):
        with pytest.raises(UserNotFoundError):
            await service.verify(
                user_id=uuid.uuid4(),
                audio_file=_make_upload_file(),
            )


@pytest.mark.asyncio
async def test_verify_raises_voiceprint_not_found():
    user_id = uuid.uuid4()
    session = AsyncMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = None

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
    ):
        with pytest.raises(VoiceprintNotFoundError):
            await service.verify(
                user_id=user_id,
                audio_file=_make_upload_file(),
            )


@pytest.mark.asyncio
async def test_verify_returns_verified_when_similarity_above_threshold():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)

    session = AsyncMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.verify_user.return_value = 0.90  # clearly above 0.65

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch.object(service, "verification_log_repo", ver_log_repo_mock),
        patch.object(service, "risk_log_repo", risk_log_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        patch(
            "app.services.verification_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/live.wav",
        ),
        patch("app.services.verification_service.cleanup_temp_file"),
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            layer1_score=0.05,
        )

    assert isinstance(result, VerificationResult)
    assert result.verified is True
    assert result.decision == DECISION_VERIFIED
    assert result.similarity_score == pytest.approx(0.90)
    assert result.risk_level == "CLEAN"


@pytest.mark.asyncio
async def test_verify_returns_mismatch_when_similarity_below_threshold():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)

    session = AsyncMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.verify_user.return_value = 0.30  # clearly below 0.65

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch.object(service, "verification_log_repo", ver_log_repo_mock),
        patch.object(service, "risk_log_repo", risk_log_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        patch(
            "app.services.verification_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/live.wav",
        ),
        patch("app.services.verification_service.cleanup_temp_file"),
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            layer1_score=0.05,
        )

    assert result.verified is False
    assert result.decision == DECISION_MISMATCH
    assert result.risk_level == "FRAUD_ALERT"


@pytest.mark.asyncio
async def test_verify_fraud_alert_when_layer1_score_high():
    """
    Even if speaker similarity is high enough, a high Layer 1 score
    should result in FRAUD_ALERT from the Risk Engine.
    """
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)

    session = AsyncMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.verify_user.return_value = 0.88  # above threshold — layer2 passes

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch.object(service, "verification_log_repo", ver_log_repo_mock),
        patch.object(service, "risk_log_repo", risk_log_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        patch(
            "app.services.verification_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/live.wav",
        ),
        patch("app.services.verification_service.cleanup_temp_file"),
    ):
        result = await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            layer1_score=0.95,  # Layer 1 flags as AI-generated
        )

    assert result.risk_level == "FRAUD_ALERT"


@pytest.mark.asyncio
async def test_verify_logs_are_persisted_on_success():
    user_id = uuid.uuid4()
    fake_vp = _make_fake_voiceprint(user_id)

    session = AsyncMock()
    service = VerificationService(session=session)

    user_repo_mock = AsyncMock()
    user_repo_mock.get_by_id.return_value = _make_fake_user(user_id)

    voiceprint_repo_mock = AsyncMock()
    voiceprint_repo_mock.get_by_user_id.return_value = fake_vp

    ver_log_repo_mock = AsyncMock()
    risk_log_repo_mock = AsyncMock()

    ecapa_mock = MagicMock()
    ecapa_mock.verify_user.return_value = 0.82

    with (
        patch.object(service, "user_repo", user_repo_mock),
        patch.object(service, "voiceprint_repo", voiceprint_repo_mock),
        patch.object(service, "verification_log_repo", ver_log_repo_mock),
        patch.object(service, "risk_log_repo", risk_log_repo_mock),
        patch.object(service, "ecapa_service", ecapa_mock),
        patch(
            "app.services.verification_service.save_upload_to_temp",
            new_callable=AsyncMock,
            return_value="/tmp/live.wav",
        ),
        patch("app.services.verification_service.cleanup_temp_file"),
    ):
        await service.verify(
            user_id=user_id,
            audio_file=_make_upload_file(),
            layer1_score=0.10,
        )

    ver_log_repo_mock.create.assert_called_once()
    risk_log_repo_mock.create.assert_called_once()
    session.commit.assert_called_once()
