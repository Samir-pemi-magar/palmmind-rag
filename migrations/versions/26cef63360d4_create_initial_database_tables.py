"""Create initial database tables.

Revision ID: 26cef63360d4
Revises:
Create Date: 2026-11-02
"""

import sqlalchemy as sa
from alembic import op

# Alembic revision identifiers.
revision = "26cef63360d4"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the initial application tables."""

    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("chunking_strategy", sa.String(length=30), nullable=False),
        sa.Column("total_chunks", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "interview_bookings",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("candidate_name", sa.String(length=150), nullable=False),
        sa.Column("candidate_email", sa.String(length=255), nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timezone", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("conversation_id", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_index(
        "ix_interview_bookings_candidate_email",
        "interview_bookings",
        ["candidate_email"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the application tables."""

    op.drop_index(
        "ix_interview_bookings_candidate_email",
        table_name="interview_bookings",
    )

    op.drop_table("interview_bookings")
    op.drop_table("documents")
