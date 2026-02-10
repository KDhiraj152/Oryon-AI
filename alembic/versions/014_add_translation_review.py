"""Add translation review system

Revision ID: 014_add_translation_review
Revises: 013_add_question_generation
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '014_add_translation_review'
down_revision = '013_add_question_generation'
branch_labels = None
depends_on = None


def upgrade():
    # Create translation_reviews table
    op.create_table(
        'translation_reviews',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('translation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('original_text', sa.Text, nullable=False),
        sa.Column('translated_text', sa.Text, nullable=False),
        sa.Column('revised_text', sa.Text, nullable=True),
        sa.Column('source_lang', sa.String(10), nullable=False),
        sa.Column('target_lang', sa.String(10), nullable=False),
        sa.Column('reviewer_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.SmallInteger, default=0, nullable=False),  # 0=pending, 1=approved, 2=rejected, 3=revised
        sa.Column('quality_score', sa.Float, nullable=True),  # 0-1 score
        sa.Column('comments', sa.Text, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['reviewer_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # Create indexes
    op.create_index('idx_translation_reviews_translation_id', 'translation_reviews', ['translation_id'])
    op.create_index('idx_translation_reviews_reviewer_id', 'translation_reviews', ['reviewer_id'])
    op.create_index('idx_translation_reviews_status', 'translation_reviews', ['status'])
    op.create_index('idx_translation_reviews_created_at', 'translation_reviews', ['created_at'])
    
    # Create review_comments table for threaded discussions
    op.create_table(
        'review_comments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('review_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('parent_comment_id', postgresql.UUID(as_uuid=True), nullable=True),  # For threading
        sa.Column('comment_text', sa.Text, nullable=False),
        sa.Column('is_resolved', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['review_id'], ['translation_reviews.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['parent_comment_id'], ['review_comments.id'], ondelete='CASCADE')
    )
    
    # Create indexes
    op.create_index('idx_review_comments_review_id', 'review_comments', ['review_id'])
    op.create_index('idx_review_comments_user_id', 'review_comments', ['user_id'])
    
    # Create review_versions table for version history
    op.create_table(
        'review_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('review_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('version_number', sa.Integer, nullable=False),
        sa.Column('revised_text', sa.Text, nullable=False),
        sa.Column('revised_by', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('change_description', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['review_id'], ['translation_reviews.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['revised_by'], ['users.id'], ondelete='CASCADE'),
        
        sa.UniqueConstraint('review_id', 'version_number', name='uq_review_version')
    )
    
    # Create indexes
    op.create_index('idx_review_versions_review_id', 'review_versions', ['review_id'])


def downgrade():
    op.drop_table('review_versions')
    op.drop_table('review_comments')
    op.drop_table('translation_reviews')
