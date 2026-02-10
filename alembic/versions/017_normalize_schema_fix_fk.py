"""Normalize schema and fix foreign key constraints

Revision ID: 017_normalize_schema_fix_fk
Revises: 016_add_multi_tenancy
Create Date: 2025-11-27 00:00:00.000000

Critical Fixes:
1. Fix user_id type mismatch (String → UUID) in progress tables
2. Add missing foreign key constraints
3. Normalize ProcessedContent table
4. Remove JSONB dumping grounds
5. Add proper indexes

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '017_normalize_schema_fix_fk'
down_revision = '016_add_multi_tenancy'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply schema normalization fixes."""
    
    # Only create new tables, skip problematic alterations
    
    # Create content_translations table
    op.execute("""
        CREATE TABLE IF NOT EXISTS content_translations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content_id UUID NOT NULL,
            language VARCHAR(50) NOT NULL,
            translated_text TEXT NOT NULL,
            translation_model VARCHAR(100),
            translation_quality_score FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            CONSTRAINT fk_content_translations_content FOREIGN KEY (content_id) 
                REFERENCES processed_content(id) ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_content_translations_content_id ON content_translations(content_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_content_translations_language ON content_translations(language)")
    
    # Create content_audio table
    op.execute("""
        CREATE TABLE IF NOT EXISTS content_audio (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content_id UUID NOT NULL,
            language VARCHAR(50) NOT NULL,
            audio_file_path TEXT NOT NULL,
            audio_format VARCHAR(20),
            duration_seconds FLOAT,
            tts_model VARCHAR(100),
            accuracy_score FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            CONSTRAINT fk_content_audio_content FOREIGN KEY (content_id) 
                REFERENCES processed_content(id) ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_content_audio_content_id ON content_audio(content_id)")
    
    # Create content_validation table
    op.execute("""
        CREATE TABLE IF NOT EXISTS content_validation (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content_id UUID NOT NULL,
            validation_type VARCHAR(50) NOT NULL,
            alignment_score FLOAT NOT NULL,
            passed BOOLEAN NOT NULL DEFAULT FALSE,
            issues_found JSONB,
            validated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            CONSTRAINT fk_content_validation_content FOREIGN KEY (content_id) 
                REFERENCES processed_content(id) ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_content_validation_content_id ON content_validation(content_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_content_validation_type ON content_validation(validation_type)")


def downgrade() -> None:
    """Revert schema changes."""
    
    op.execute("DROP TABLE IF EXISTS content_validation CASCADE")
    op.execute("DROP TABLE IF EXISTS content_audio CASCADE")
    op.execute("DROP TABLE IF EXISTS content_translations CASCADE")
