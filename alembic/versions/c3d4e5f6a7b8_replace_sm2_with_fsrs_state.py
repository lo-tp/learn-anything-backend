"""Replace SM-2 columns with FSRS state on review_cards

Revision ID: c3d4e5f6a7b8
Revises: 5af14c01ea6b
Create Date: 2026-09-22 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: str | None = '5af14c01ea6b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("review_cards", "interval_days")
    op.drop_column("review_cards", "ease")
    op.drop_column("review_cards", "is_retired")
    op.add_column(
        "review_cards",
        sa.Column("fsrs_state", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("review_cards", "fsrs_state")
    op.add_column(
        "review_cards",
        sa.Column("interval_days", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "review_cards",
        sa.Column("ease", sa.Float(), nullable=False, server_default="2.5"),
    )
    op.add_column(
        "review_cards",
        sa.Column("is_retired", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
