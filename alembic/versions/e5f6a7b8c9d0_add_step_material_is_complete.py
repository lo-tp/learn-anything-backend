"""add step_materials.is_complete

Revision ID: e5f6a7b8c9d0
Revises: dccdd4b6d993
Create Date: 2026-09-16 11:30:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: str | None = 'dccdd4b6d993'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing rows were only ever written when a step completed, so they are
    # all complete. Backfill to True so a re-run does not treat them as
    # provisional (which would re-generate finished steps).
    op.add_column(
        'step_materials',
        sa.Column('is_complete', sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    # Drop the server default so the model default (False) governs new rows —
    # a fresh insert is provisional until explicitly finalized.
    op.alter_column('step_materials', 'is_complete', server_default=None)


def downgrade() -> None:
    op.drop_column('step_materials', 'is_complete')
