"""make timestamps timezone-aware

Revision ID: 920b71d3527b
Revises: fda7a4fdf4
Create Date: 2026-09-17 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '920b71d3527b'
down_revision: str | None = 'fda7a4fdf4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, column, nullable) pairs for every datetime column in the schema.
COLUMNS: list[tuple[str, str, bool]] = [
    ('sessions', 'created_at', False),
    ('sessions', 'updated_at', False),
    ('probe_questions', 'answered_at', True),
    ('failed_slides', 'created_at', False),
    ('graph_stage_timings', 'created_at', False),
]


def upgrade() -> None:
    # All writes are UTC wall-clock (datetime.now(UTC)), so converting the
    # naive values to timestamptz under the server's UTC TimeZone preserves
    # the exact instants; responses then carry an explicit offset.
    for table, column, nullable in COLUMNS:
        op.alter_column(
            table,
            column,
            existing_type=sa.DateTime(),
            type_=sa.DateTime(timezone=True),
            nullable=nullable,
        )


def downgrade() -> None:
    for table, column, nullable in COLUMNS:
        op.alter_column(
            table,
            column,
            existing_type=sa.DateTime(timezone=True),
            type_=sa.DateTime(),
            nullable=nullable,
        )
