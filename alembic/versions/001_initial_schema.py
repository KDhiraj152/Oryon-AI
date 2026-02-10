"""Initial schema for education content pipeline.

Revision ID: 001
Revises: 
Create Date: 2025-11-10

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create processed_content table
    op.create_table(
        'processed_content',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('original_text', sa.Text(), nullable=False),
        sa.Column('simplified_text', sa.Text()),
        sa.Column('translated_text', sa.Text()),
        sa.Column('language', sa.String(50), nullable=False),
        sa.Column('grade_level', sa.Integer(), nullable=False),
        sa.Column('subject', sa.String(100), nullable=False),
        sa.Column('audio_file_path', sa.Text()),
        sa.Column('ncert_alignment_score', sa.Float()),
        sa.Column('audio_accuracy_score', sa.Float()),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now()),
        sa.Column('metadata', JSONB())
    )
    
    # Create indexes for common queries
    op.create_index('idx_content_language', 'processed_content', ['language'])
    op.create_index('idx_content_grade', 'processed_content', ['grade_level'])
    op.create_index('idx_content_subject', 'processed_content', ['subject'])
    
    # Create ncert_standards table
    op.create_table(
        'ncert_standards',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('grade_level', sa.Integer(), nullable=False),
        sa.Column('subject', sa.String(100), nullable=False),
        sa.Column('topic', sa.String(200), nullable=False),
        sa.Column('learning_objectives', ARRAY(sa.Text())),
        sa.Column('keywords', ARRAY(sa.Text()))
    )
    
    # Create indexes for NCERT standards lookup
    op.create_index('idx_ncert_grade_subject', 'ncert_standards', ['grade_level', 'subject'])
    
    # Create student_profiles table
    op.create_table(
        'student_profiles',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('language_preference', sa.String(50), nullable=False),
        sa.Column('grade_level', sa.Integer(), nullable=False),
        sa.Column('subjects_of_interest', ARRAY(sa.Text())),
        sa.Column('offline_content_cache', JSONB()),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Create pipeline_logs table
    op.create_table(
        'pipeline_logs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('content_id', UUID(as_uuid=True)),
        sa.Column('stage', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('processing_time_ms', sa.Integer()),
        sa.Column('error_message', sa.Text()),
        sa.Column('timestamp', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Create foreign key constraint
    op.create_foreign_key(
        'fk_pipeline_logs_content',
        'pipeline_logs', 'processed_content',
        ['content_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Create indexes for pipeline logs
    op.create_index('idx_logs_content_id', 'pipeline_logs', ['content_id'])
    op.create_index('idx_logs_stage', 'pipeline_logs', ['stage'])
    op.create_index('idx_logs_timestamp', 'pipeline_logs', ['timestamp'])


def downgrade() -> None:
    op.drop_table('pipeline_logs')
    op.drop_table('student_profiles')
    op.drop_table('ncert_standards')
    op.drop_table('processed_content')
