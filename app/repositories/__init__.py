"""
app/repositories/__init__.py

Repository layer: encapsulates all direct database access (SQLAlchemy
queries) behind simple, typed methods. Services depend on repositories,
never on raw SQLAlchemy sessions/queries directly, keeping persistence
concerns isolated and easy to test/mock.
"""

from app.repositories.user_repository import UserRepository
from app.repositories.voiceprint_repository import VoiceprintRepository
from app.repositories.verification_repository import VerificationLogRepository
from app.repositories.risk_repository import RiskLogRepository

__all__ = [
    "UserRepository",
    "VoiceprintRepository",
    "VerificationLogRepository",
    "RiskLogRepository",
]
