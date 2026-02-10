"""Add composite indexes to processed_content table

Revision ID: 005_add_composite_indexes
Revises: 61631d311ed9
Create Date: 2025-11-16

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '005_add_composite_indexes'
down_revision = '004_add_user_tracking'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add composite indexes for common query patterns."""
    # Create composite indexes
    op.create_index(
        'idx_user_content',
        'processed_content',
        ['user_id', 'created_at'],
        unique=False
    )
    op.create_index(
        'idx_grade_subject',
        'processed_content',
        ['grade_level', 'subject'],
        unique=False
    )
    op.create_index(
        'idx_language_grade',
        'processed_content',
        ['language', 'grade_level'],
        unique=False
    )
    op.create_index(
        'idx_subject_created',
        'processed_content',
        ['subject', 'created_at'],
        unique=False
    )


def downgrade() -> None:
    """Remove composite indexes."""
    op.drop_index('idx_subject_created', table_name='processed_content')
    op.drop_index('idx_language_grade', table_name='processed_content')
    op.drop_index('idx_grade_subject', table_name='processed_content')
    op.drop_index('idx_user_content', table_name='processed_content')
