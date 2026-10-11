"""add plan_preferences table

Preference pairs captured when a learner approves a plan they had adjusted
(#167): the prompt the refinement was conditioned on, the rejected version,
the chosen version, and the adjustment text. Nothing in the serving path
reads it — it is offline training data.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-10-11 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: str | None = 'c3d4e5f6a7b8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "plan_preferences",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("rejected_version", sa.Integer(), nullable=False),
        sa.Column("chosen_version", sa.Integer(), nullable=False),
        sa.Column("prompt_context", sa.JSON(), nullable=False),
        sa.Column("adjustment", sa.Text(), nullable=False),
        sa.Column("rejected_plan", sa.JSON(), nullable=False),
        sa.Column("chosen_plan", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["sessions.session_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_plan_preferences_session_id", "plan_preferences", ["session_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_plan_preferences_session_id", table_name="plan_preferences")
    op.drop_table("plan_preferences")
