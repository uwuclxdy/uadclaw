"""add fact extraction tables

Revision ID: fe79b328d598
Revises: b41f7c0a9e15
Create Date: 2026-08-11 15:33:27.339754

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "fe79b328d598"
down_revision: str | Sequence[str] | None = "b41f7c0a9e15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "package_facts",
        sa.Column("package", sa.String(length=255), nullable=False),
        sa.Column("device_count", sa.Integer(), nullable=False),
        sa.Column("devices", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("label_unresolved", sa.Boolean(), nullable=False),
        sa.Column("version_code", sa.BigInteger(), nullable=True),
        sa.Column("partitions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("priv_app", sa.Boolean(), nullable=False),
        sa.Column("cert_issuer", sa.Text(), nullable=True),
        sa.Column("cert_subject", sa.Text(), nullable=True),
        sa.Column("core_app", sa.Boolean(), nullable=False),
        sa.Column("shared_user_id", sa.String(length=255), nullable=True),
        sa.Column("persistent", sa.Boolean(), nullable=False),
        sa.Column("has_code", sa.Boolean(), nullable=False),
        sa.Column("overlay_target", sa.String(length=255), nullable=True),
        sa.Column("overlay_static", sa.Boolean(), nullable=False),
        sa.Column("is_input_method", sa.Boolean(), nullable=False),
        sa.Column("is_device_admin", sa.Boolean(), nullable=False),
        sa.Column("is_accessibility_service", sa.Boolean(), nullable=False),
        sa.Column("is_carrier_service", sa.Boolean(), nullable=False),
        sa.Column("libraries", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("static_libraries", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "uses_libraries_required", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "uses_libraries_optional", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("protected_broadcasts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("provider_authorities", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("intent_filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("has_conflict", sa.Boolean(), nullable=False),
        sa.Column(
            "conflicts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("package"),
    )
    op.create_index(
        "ix_package_facts_device_count", "package_facts", ["device_count"], unique=False
    )
    op.create_index(
        "ix_package_facts_has_conflict", "package_facts", ["has_conflict"], unique=False
    )
    op.create_table(
        "package_observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("device_key", sa.String(length=255), nullable=False),
        sa.Column("build", sa.String(length=255), nullable=False),
        sa.Column("package", sa.String(length=255), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("label_unresolved", sa.Boolean(), nullable=False),
        sa.Column("version_code", sa.BigInteger(), nullable=True),
        sa.Column("partition", sa.String(length=64), nullable=False),
        sa.Column("device_path", sa.Text(), nullable=False),
        sa.Column("priv_app", sa.Boolean(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("cert_issuer", sa.Text(), nullable=True),
        sa.Column("cert_subject", sa.Text(), nullable=True),
        sa.Column("core_app", sa.Boolean(), nullable=False),
        sa.Column("shared_user_id", sa.String(length=255), nullable=True),
        sa.Column("persistent", sa.Boolean(), nullable=False),
        sa.Column("has_code", sa.Boolean(), nullable=False),
        sa.Column("overlay_target", sa.String(length=255), nullable=True),
        sa.Column("overlay_static", sa.Boolean(), nullable=False),
        sa.Column("overlay_priority", sa.Integer(), nullable=True),
        sa.Column("is_input_method", sa.Boolean(), nullable=False),
        sa.Column("is_device_admin", sa.Boolean(), nullable=False),
        sa.Column("is_accessibility_service", sa.Boolean(), nullable=False),
        sa.Column("is_carrier_service", sa.Boolean(), nullable=False),
        sa.Column("libraries", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("static_libraries", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "uses_libraries_required", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "uses_libraries_optional", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("protected_broadcasts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("provider_authorities", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("intent_filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "device_key", "build", "device_path", name="uq_package_observations_device_path"
        ),
    )
    op.create_index(
        "ix_package_observations_package", "package_observations", ["package"], unique=False
    )
    op.create_table(
        "device_scans",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=True),
        sa.Column("device_key", sa.String(length=255), nullable=False),
        sa.Column("build", sa.String(length=255), nullable=False),
        sa.Column("scanned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("apk_total", sa.Integer(), nullable=False),
        sa.Column("parsed_ok", sa.Integer(), nullable=False),
        sa.Column("parse_failed", sa.Integer(), nullable=False),
        sa.Column(
            "failures",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_device_scans_device_key", "device_scans", ["device_key"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_device_scans_device_key", table_name="device_scans")
    op.drop_table("device_scans")
    op.drop_index("ix_package_observations_package", table_name="package_observations")
    op.drop_table("package_observations")
    op.drop_index("ix_package_facts_has_conflict", table_name="package_facts")
    op.drop_index("ix_package_facts_device_count", table_name="package_facts")
    op.drop_table("package_facts")
