"""
tests/test_risk_engine.py

Unit tests for app.services.risk_engine.compute_risk, covering the three
documented decision rules:

    1. layer1_score > LAYER1_FRAUD_THRESHOLD (0.70)        -> FRAUD_ALERT
    2. speaker_similarity < SIMILARITY_THRESHOLD (0.65)    -> FRAUD_ALERT
    3. otherwise                                           -> CLEAN
"""

from app.core.config import get_settings
from app.services.risk_engine import CLEAN, FRAUD_ALERT, compute_risk

settings = get_settings()


def test_layer1_above_threshold_triggers_fraud_alert():
    result = compute_risk(layer1_score=0.85, speaker_similarity=0.95)
    assert result.risk_level == FRAUD_ALERT


def test_low_similarity_triggers_fraud_alert():
    result = compute_risk(layer1_score=0.05, speaker_similarity=0.40)
    assert result.risk_level == FRAUD_ALERT


def test_clean_when_layer1_low_and_similarity_high():
    result = compute_risk(layer1_score=0.05, speaker_similarity=0.90)
    assert result.risk_level == CLEAN


def test_layer1_exactly_at_threshold_is_not_fraud_by_layer1_alone():
    # layer1_score == 0.70 is NOT strictly greater than the threshold,
    # so rule 1 should not fire on its own.
    result = compute_risk(layer1_score=0.70, speaker_similarity=0.90)
    assert result.risk_level == CLEAN


def test_similarity_exactly_at_threshold_is_not_fraud_by_similarity_alone():
    # speaker_similarity == 0.65 is NOT strictly less than the threshold,
    # so rule 2 should not fire on its own.
    result = compute_risk(layer1_score=0.05, speaker_similarity=0.65)
    assert result.risk_level == CLEAN


def test_both_rules_triggered_still_results_in_fraud_alert():
    result = compute_risk(layer1_score=0.95, speaker_similarity=0.10)
    assert result.risk_level == FRAUD_ALERT


def test_risk_score_is_bounded_between_0_and_100():
    for layer1_score, similarity in [
        (0.0, 1.0),
        (1.0, -1.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (0.0, -1.0),
    ]:
        result = compute_risk(layer1_score, similarity)
        assert 0.0 <= result.risk_score <= 100.0


def test_perfect_match_and_zero_ai_probability_yields_zero_risk_score():
    result = compute_risk(layer1_score=0.0, speaker_similarity=1.0)
    assert result.risk_score == 0.0
    assert result.risk_level == CLEAN


def test_worst_case_yields_max_risk_score():
    # layer1_score = 1.0 (certainly AI) and similarity = -1.0 (total mismatch)
    result = compute_risk(layer1_score=1.0, speaker_similarity=-1.0)
    assert result.risk_score == 100.0
    assert result.risk_level == FRAUD_ALERT
