"""add slide_contents.is_regenerating

Revision ID: 9c0d1e2f3a4b
Revises: 5d4742b10d4a
Create Date: 2026-10-15 09:40:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9c0d1e2f3a4b'
down_revision: str | None = '5d4742b10d4a'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # True while a single-slide regeneration of this row is running (#197),
    # surfaced as `regenerating` on the slide item of
    # GET /sessions/{session_id}/materials — docs/api-and-graphs.md states why a
    # rewritten slide needs a signal of its own.
    #
    # Existing rows are not being rewritten, so backfill to False; the model
    # default governs new ones. Nothing needs a backfill of *truth*: a rewrite is
    # a task inside a running process, and app startup clears whatever a previous
    # process left behind (routers.plan.clear_abandoned_regenerations).
    op.add_column(
        'slide_contents',
        sa.Column('is_regenerating', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column('slide_contents', 'is_regenerating', server_default=None)


def downgrade() -> None:
    op.drop_column('slide_contents', 'is_regenerating')
