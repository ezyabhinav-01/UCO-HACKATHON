"""
tests/test_ecapa_service.py

Unit tests for app.ml.ecapa_service.ECAPAService.cosine_similarity and
ECAPAService.enroll_user's averaging logic.

These tests do NOT load the actual ECAPA-TDNN model (which requires a
~100MB download), so they exercise the pure-math helper functions and the
averaging/aggregation logic using mocked embeddings.
"""

from unittest.mock import patch

import numpy as np
import pytest

from app.ml.ecapa_service import ECAPAService


def test_cosine_similarity_identical_vectors_is_one():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert ECAPAService.cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert ECAPAService.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-1.0, -2.0, -3.0])
    assert ECAPAService.cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_handles_zero_vector_gracefully():
    a = np.zeros(192)
    b = np.random.rand(192)
    assert ECAPAService.cosine_similarity(a, b) == 0.0


def test_cosine_similarity_is_scale_invariant():
    a = np.array([1.0, 2.0, 3.0])
    b = a * 100.0
    assert ECAPAService.cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_clamped_to_valid_range():
    # Construct vectors whose dot product could, due to floating point
    # error, slightly exceed [-1, 1]. The implementation should clamp.
    a = np.array([1.0, 1.0, 1.0]) * 1e10
    b = np.array([1.0, 1.0, 1.0]) * 1e10
    score = ECAPAService.cosine_similarity(a, b)
    assert -1.0 <= score <= 1.0


def test_enroll_user_averages_embeddings_correctly():
    service = ECAPAService()

    fake_embeddings = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([3.0, 4.0, 5.0], dtype=np.float32),
        np.array([2.0, 0.0, 1.0], dtype=np.float32),
    ]

    with patch.object(service, "extract_embedding", side_effect=fake_embeddings):
        averaged, count = service.enroll_user(["a.wav", "b.wav", "c.wav"])

    expected = np.array([2.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(averaged, expected)
    assert count == 3


def test_enroll_user_skips_failed_files():
    from app.utils.exceptions import EmbeddingExtractionError

    service = ECAPAService()

    good_embedding = np.array([1.0, 1.0], dtype=np.float32)

    def side_effect(path):
        if path == "bad.wav":
            raise EmbeddingExtractionError(path, "corrupt file")
        return good_embedding

    with patch.object(service, "extract_embedding", side_effect=side_effect):
        averaged, count = service.enroll_user(["good1.wav", "bad.wav", "good2.wav"])

    np.testing.assert_allclose(averaged, good_embedding)
    assert count == 2


def test_enroll_user_returns_zero_vector_when_all_files_fail():
    from app.core.config import get_settings
    from app.utils.exceptions import EmbeddingExtractionError

    settings = get_settings()
    service = ECAPAService()

    def side_effect(path):
        raise EmbeddingExtractionError(path, "corrupt file")

    with patch.object(service, "extract_embedding", side_effect=side_effect):
        averaged, count = service.enroll_user(["bad1.wav", "bad2.wav"])

    assert count == 0
    assert averaged.shape == (settings.EMBEDDING_DIM,)
    assert np.all(averaged == 0.0)
