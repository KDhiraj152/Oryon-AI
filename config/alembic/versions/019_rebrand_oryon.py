"""Rebrand: Rename database from shiksha_setu to oryon

Revision ID: 019_rebrand_oryon
Revises: 018_add_chat_conversations
Create Date: 2026-02-16 00:00:00.000000

This migration handles the brand transition from Shiksha Setu to Oryon AI.
It renames any database-level artifacts that referenced the old brand.

NOTE: Database-level rename (ALTER DATABASE) must be done outside Alembic
since you cannot rename the database you're connected to. Run this manually:

    -- Connect to postgres (not the app database)
    -- ALTER DATABASE shiksha_setu RENAME TO oryon;
    -- Then update DATABASE_URL to point to 'oryon'

This migration handles schema-level changes only.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '019_rebrand_oryon'
down_revision = '018_add_chat_conversations'
branch_labels = None
depends_on = None


def upgrade():
    """Rebrand: Update any constraints, indexes, or comments referencing old brand.

    The actual table/column structure does not use brand-prefixed names,
    so this migration serves as a checkpoint and documentation of the rename.

    If your deployment had brand-prefixed table names, add ALTER TABLE
    statements here. Example:

        op.rename_table('shiksha_setu_users', 'oryon_users')
        op.rename_table('shiksha_setu_sessions', 'oryon_sessions')
    """
    # Update database comment to reflect new brand
    op.execute(
        sa.text("COMMENT ON DATABASE CURRENT_DATABASE IS 'Oryon AI Platform Database'")
    ) if _supports_comments() else None

    # Add a metadata marker for the rebrand
    op.execute(
        sa.text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'oryon_metadata'
                ) THEN
                    CREATE TABLE oryon_metadata (
                        key VARCHAR(255) PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                END IF;
            END $$;
        """)
    )
    op.execute(
        sa.text("""
            INSERT INTO oryon_metadata (key, value)
            VALUES ('brand_version', 'oryon-ai-v1.0')
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
        """)
    )
    op.execute(
        sa.text("""
            INSERT INTO oryon_metadata (key, value)
            VALUES ('rebrand_date', NOW()::text)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
        """)
    )


def downgrade():
    """Revert rebrand metadata."""
    op.execute(sa.text("DROP TABLE IF EXISTS oryon_metadata;"))


def _supports_comments():
    """Check if the database backend supports COMMENT ON DATABASE."""
    try:
        bind = op.get_bind()
        return bind.dialect.name == 'postgresql'
    except Exception:
        return False
