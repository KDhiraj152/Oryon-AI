"""Add generated_questions table for AI question generation.

Revision ID: 013_add_question_generation
Revises: 012_add_hnsw_indexes
Create Date: 2024-01-16 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '013_add_question_generation'
down_revision = '012_add_hnsw_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create generated_questions table."""
    
    op.create_table(
        'generated_questions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('processed_content.id'), nullable=False),
        sa.Column('question_text', sa.Text(), nullable=False),
        sa.Column('question_type', sa.String(50), nullable=False),
        sa.Column('options', postgresql.JSON(), nullable=True),
        sa.Column('correct_answer', sa.Text(), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('difficulty', sa.String(20), nullable=False),
        sa.Column('content_objective', sa.String(200), nullable=True),
        sa.Column('bloom_taxonomy_level', sa.String(50), nullable=True),
        sa.Column('quality_score', sa.Integer(), default=0),
        sa.Column('is_approved', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('metadata', postgresql.JSON(), default=dict),
    )
    
    # Create indexes
    op.create_index('idx_questions_document_id', 'generated_questions', ['document_id'])
    op.create_index('idx_questions_approval_status', 'generated_questions', ['is_approved'])
    op.create_index('idx_questions_difficulty', 'generated_questions', ['difficulty'])
    op.create_index('idx_questions_type', 'generated_questions', ['question_type'])
    
    print("✓ generated_questions table created")
    print("  - Supports MCQ, short answer, true/false questions")
    print("  - Content objective tagging")
    print("  - Bloom's taxonomy classification")
    print("  - Quality scoring and approval workflow")


def downgrade() -> None:
    """Drop generated_questions table."""
    
    op.drop_index('idx_questions_type', 'generated_questions')
    op.drop_index('idx_questions_difficulty', 'generated_questions')
    op.drop_index('idx_questions_approval_status', 'generated_questions')
    op.drop_index('idx_questions_document_id', 'generated_questions')
    op.drop_table('generated_questions')
    
    print("✓ generated_questions table dropped")
