"""create initial layer 2 schema (users, voiceprints, verification_logs, risk_logs)

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-15 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 192


def upgrade() -> None:
    # Required for the VECTOR column type used by `voiceprints.embedding`.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------
    # users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ------------------------------------------------------------------
    # voiceprints
    # ------------------------------------------------------------------
    op.create_table(
        "voiceprints",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "recording_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_voiceprints_user_id", "voiceprints", ["user_id"], unique=True
    )

    # ------------------------------------------------------------------
    # verification_logs
    # ------------------------------------------------------------------
    op.create_table(
        "verification_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_verification_logs_user_id", "verification_logs", ["user_id"]
    )

    # ------------------------------------------------------------------
    # risk_logs
    # ------------------------------------------------------------------
    op.create_table(
        "risk_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_risk_logs_user_id", "risk_logs", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_risk_logs_user_id", table_name="risk_logs")
    op.drop_table("risk_logs")

    op.drop_index("ix_verification_logs_user_id", table_name="verification_logs")
    op.drop_table("verification_logs")

    op.drop_index("ix_voiceprints_user_id", table_name="voiceprints")
    op.drop_table("voiceprints")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
