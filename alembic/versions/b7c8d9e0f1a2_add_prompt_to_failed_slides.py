"""add prompt to failed_slides

Revision ID: b7c8d9e0f1a2
Revises: e5f6a7b8c9d0
Create Date: 2026-09-16 14:05:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7c8d9e0f1a2'
down_revision: str | None = 'e5f6a7b8c9d0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # failed_slides already stores jsx (LLM result) and error (compile error)
    # per failed attempt. Add the prompt (what we sent the LLM) so each row is
    # a complete debug record: prompt -> result -> error.
    # The NOT NULL default backfills legacy rows with ''; the default is then
    # dropped so future rows must always carry a real prompt.
    op.add_column(
        'failed_slides',
        sa.Column('prompt', sa.Text(), nullable=False, server_default=''),
    )
    op.alter_column('failed_slides', 'prompt', server_default=None)


def downgrade() -> None:
    op.drop_column('failed_slides', 'prompt')
