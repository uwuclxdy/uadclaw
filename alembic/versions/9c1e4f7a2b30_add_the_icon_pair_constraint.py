"""add the icon pair constraint

Revision ID: 9c1e4f7a2b30
Revises: 37b785e1bcb4
Create Date: 2026-08-12 22:05:00.000000

Its own revision rather than an edit to 37b785e1bcb4: that revision has already been applied to
every per-worker test database on this box, and a database sitting at head never re-runs a
revision whose CONTENT changed, so amending it would leave those databases without the
constraint while reading as up to date.

Written by hand: alembic's autogenerate does not compare CHECK constraints, so `alembic check`
is clean with or without this and cannot be the thing that catches its absence.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c1e4f7a2b30"
down_revision: str | Sequence[str] | None = "37b785e1bcb4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_check_constraint(
        "ck_package_facts_icon_pair", "package_facts", "(icon_bytes IS NULL) = (icon_mime IS NULL)"
    )
    op.create_check_constraint(
        "ck_package_observations_icon_pair",
        "package_observations",
        "(icon_bytes IS NULL) = (icon_mime IS NULL)",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_package_observations_icon_pair", "package_observations")
    op.drop_constraint("ck_package_facts_icon_pair", "package_facts")
