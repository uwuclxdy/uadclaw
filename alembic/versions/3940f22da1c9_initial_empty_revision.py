"""initial empty revision

Revision ID: 3940f22da1c9
Revises:
Create Date: 2026-08-11 10:49:31.303449

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "3940f22da1c9"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
