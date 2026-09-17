"""add ON DELETE CASCADE to session FKs

Revision ID: fda7a4fdf4
Revises: 80218d7f21
Create Date: 2026-09-17 10:25:05
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'fda7a4fdf4'
down_revision: str | None = '80218d7f21'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# All tables whose session FK used the default NO ACTION. Deleting a
# session now removes all of its children.
_TABLES = (
    'plans',
    'probe_questions',
    'slide_contents',
    'step_materials',
    'step_progress',
    'failed_slides',
    'graph_stage_timings',
)


def upgrade() -> None:
    for table in _TABLES:
        name = f'{table}_session_id_fkey'
        op.drop_constraint(name, table_name=table)
        op.create_foreign_key(
            name, table, 'sessions', ['session_id'], ['session_id'],
            ondelete='CASCADE',
        )


def downgrade() -> None:
    for table in _TABLES:
        name = f'{table}_session_id_fkey'
        op.drop_constraint(name, table_name=table)
        op.create_foreign_key(
            name, table, 'sessions', ['session_id'], ['session_id'],
        )
