"""add session language

Revision ID: f3e2d1c0b9a8
Revises: 3ff84fb69b6f
Create Date: 2026-09-13 10:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f3e2d1c0b9a8'
down_revision: str | None = '3ff84fb69b6f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'sessions',
        sa.Column('language', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('sessions', 'language')
