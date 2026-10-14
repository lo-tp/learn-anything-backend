"""add slide_contents.content_spec

Revision ID: 5d4742b10d4a
Revises: c3d4e5f6a7b8
Create Date: 2026-10-14 16:20:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '5d4742b10d4a'
down_revision: str | None = 'c3d4e5f6a7b8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The slide content spec ({title, key_points, visual_hint}) that
    # plan_slide_contents produced for this slide, stored beside the JSX it
    # produced (#197). Every other input to write_slide is already
    # reconstructable from this database; this was the one that was not, which
    # is why regenerating a single slide was impossible.
    #
    # Nullable and deliberately not backfilled: a row written before this column
    # has no spec to recover — plan_slide_contents' output was never written
    # anywhere but the graph's state. Those slides keep the whole-deck
    # regeneration route.
    op.add_column(
        'slide_contents',
        sa.Column('content_spec', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('slide_contents', 'content_spec')
