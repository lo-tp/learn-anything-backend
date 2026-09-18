"""add slide_contents.is_placeholder

Revision ID: a1b2c3d4e5f6
Revises: 920b71d3527b
Create Date: 2026-09-18 11:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: str | None = '920b71d3527b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing rows are all real (non-placeholder) slides, so backfill to
    # False — the model default governs new rows.
    op.add_column(
        'slide_contents',
        sa.Column('is_placeholder', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column('slide_contents', 'is_placeholder', server_default=None)


def downgrade() -> None:
    op.drop_column('slide_contents', 'is_placeholder')
