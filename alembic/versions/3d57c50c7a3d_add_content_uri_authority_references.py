"""add content uri authority references

Revision ID: 3d57c50c7a3d
Revises: 60bbce12a008
Create Date: 2026-08-14 10:44:20.674797

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "3d57c50c7a3d"
down_revision: str | Sequence[str] | None = "60bbce12a008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "package_observations",
        sa.Column(
            "content_uri_authorities",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "package_facts",
        sa.Column(
            "content_uri_authorities",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("package_facts", "content_uri_authorities")
    op.drop_column("package_observations", "content_uri_authorities")
