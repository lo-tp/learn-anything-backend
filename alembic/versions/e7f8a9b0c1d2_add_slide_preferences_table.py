"""add slide_preferences table

Compiler-annotated preference pairs (#168): one row per slide whose JSX failed
to compile and was then fixed. The sandbox is the annotator, so the pair is
free — the prompt context, the rejected JSX with the error that rejected it,
and the JSX that compiled (with the prompt each half was actually written from,
so a reader never has to reconstruct one). Nothing in the serving path reads
it; it is offline preference-training data.

Revision ID: e7f8a9b0c1d2
Revises: d4e5f6a7b8c9
Create Date: 2026-10-11 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e7f8a9b0c1d2'
down_revision: str | None = 'd4e5f6a7b8c9'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "slide_preferences",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("step_id", sa.String(), nullable=False),
        sa.Column("slide_index", sa.Integer(), nullable=False),
        sa.Column("rejected_attempt", sa.Integer(), nullable=False),
        sa.Column("chosen_attempt", sa.Integer(), nullable=False),
        sa.Column("prompt_context", sa.Text(), nullable=False),
        sa.Column("chosen_prompt", sa.Text(), nullable=False),
        sa.Column("rejected_jsx", sa.Text(), nullable=False),
        sa.Column("rejected_error", sa.Text(), nullable=False),
        sa.Column("chosen_jsx", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["sessions.session_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_slide_preferences_session_id", "slide_preferences", ["session_id"]
    )


def downgrade() -> None:
    op.drop_index(
        "ix_slide_preferences_session_id", table_name="slide_preferences"
    )
    op.drop_table("slide_preferences")
