"""Add token rotation and teacher evaluation models

Revision ID: 011_token_rotation_teacher_eval
Revises: 010_upgrade_embeddings
Create Date: 2025-11-27 01:00:00.000000

Changes:
1. Add token_blacklist table for logout/rotation tracking
2. Add refresh_tokens table for session management
3. Add teacher_profiles table
4. Add content_quality_metrics table
5. Add student_engagement_metrics table
6. Add learning_outcome_metrics table
7. Add teacher_evaluations table

Issues Fixed:
- CODE-REVIEW-GPT #6 (CRITICAL): Token rotation mechanism
- CODE-REVIEW-GPT #2 (CRITICAL): PS4 Teacher evaluation foundation
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '011_token_rotation_teacher_eval'
down_revision = '010_upgrade_embeddings'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply schema changes."""
    
    # ========================================================================
    # TOKEN ROTATION TABLES
    # ========================================================================
    
    # Token blacklist table
    op.create_table(
        'token_blacklist',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('token_jti', sa.String(255), unique=True, nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('reason', sa.String(100), nullable=False),
        sa.Column('blacklisted_at', sa.TIMESTAMP, server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    op.create_index('ix_token_blacklist_jti', 'token_blacklist', ['token_jti'])
    op.create_index('ix_token_blacklist_user', 'token_blacklist', ['user_id'])
    op.create_index('ix_token_blacklist_expires', 'token_blacklist', ['expires_at'])
    
    # Refresh tokens table
    op.create_table(
        'refresh_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('token_jti', sa.String(255), unique=True, nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('device_fingerprint', sa.String(255)),
        sa.Column('ip_address', sa.String(50)),
        sa.Column('user_agent', sa.Text),
        sa.Column('parent_jti', sa.String(255)),
        sa.Column('rotation_count', sa.Integer, default=0),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False),
        sa.Column('last_used_at', sa.TIMESTAMP, server_default=sa.text('now()')),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    op.create_index('ix_refresh_tokens_jti', 'refresh_tokens', ['token_jti'])
    op.create_index('ix_refresh_tokens_user', 'refresh_tokens', ['user_id'])
    op.create_index('ix_refresh_tokens_device', 'refresh_tokens', ['device_fingerprint'])
    op.create_index('ix_refresh_tokens_parent', 'refresh_tokens', ['parent_jti'])
    
    # ========================================================================
    # TEACHER EVALUATION TABLES
    # ========================================================================
    
    # Teacher profiles
    op.create_table(
        'teacher_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), unique=True, nullable=False),
        sa.Column('employee_id', sa.String(100), unique=True),
        sa.Column('qualification', sa.String(255)),
        sa.Column('specialization', sa.String(255)),
        sa.Column('experience_years', sa.Float),
        sa.Column('subjects', postgresql.JSONB, default=[]),
        sa.Column('grade_levels', postgresql.JSONB, default=[]),
        sa.Column('overall_rating', sa.Float, default=0.0),
        sa.Column('content_quality_score', sa.Float, default=0.0),
        sa.Column('student_engagement_score', sa.Float, default=0.0),
        sa.Column('learning_outcome_score', sa.Float, default=0.0),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.Column('last_evaluated_at', sa.TIMESTAMP),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    op.create_index('ix_teacher_profiles_user', 'teacher_profiles', ['user_id'])
    op.create_index('ix_teacher_profiles_rating', 'teacher_profiles', ['overall_rating'])
    op.create_index('ix_teacher_profiles_subjects', 'teacher_profiles', ['subjects'], postgresql_using='gin')
    
    # Content quality metrics
    op.create_table(
        'content_quality_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('teacher_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('ncert_alignment_score', sa.Float, nullable=False),
        sa.Column('language_clarity_score', sa.Float, nullable=False),
        sa.Column('pedagogical_effectiveness', sa.Float, nullable=False),
        sa.Column('cultural_sensitivity_score', sa.Float, nullable=False),
        sa.Column('views_count', sa.Integer, default=0),
        sa.Column('completion_rate', sa.Float, default=0.0),
        sa.Column('average_time_spent', sa.Float, default=0.0),
        sa.Column('positive_feedback_count', sa.Integer, default=0),
        sa.Column('negative_feedback_count', sa.Integer, default=0),
        sa.Column('average_rating', sa.Float, default=0.0),
        sa.Column('overall_quality_score', sa.Float, nullable=False),
        sa.Column('evaluated_at', sa.TIMESTAMP, server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['teacher_id'], ['teacher_profiles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['content_id'], ['processed_content.id'], ondelete='CASCADE')
    )
    op.create_index('ix_quality_teacher', 'content_quality_metrics', ['teacher_id'])
    op.create_index('ix_quality_content', 'content_quality_metrics', ['content_id'])
    op.create_index('ix_quality_score', 'content_quality_metrics', ['overall_quality_score'])
    op.create_index('ix_quality_teacher_date', 'content_quality_metrics', ['teacher_id', 'evaluated_at'])
    
    # Student engagement metrics
    op.create_table(
        'student_engagement_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('teacher_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('total_students', sa.Integer, default=0),
        sa.Column('active_students', sa.Integer, default=0),
        sa.Column('completion_count', sa.Integer, default=0),
        sa.Column('dropout_count', sa.Integer, default=0),
        sa.Column('question_asked_count', sa.Integer, default=0),
        sa.Column('discussion_participation', sa.Integer, default=0),
        sa.Column('resource_download_count', sa.Integer, default=0),
        sa.Column('average_session_duration', sa.Float, default=0.0),
        sa.Column('total_learning_time', sa.Float, default=0.0),
        sa.Column('engagement_score', sa.Float, nullable=False),
        sa.Column('period_start', sa.TIMESTAMP, nullable=False),
        sa.Column('period_end', sa.TIMESTAMP, nullable=False),
        sa.Column('calculated_at', sa.TIMESTAMP, server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['teacher_id'], ['teacher_profiles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['content_id'], ['processed_content.id'], ondelete='CASCADE')
    )
    op.create_index('ix_engagement_teacher', 'student_engagement_metrics', ['teacher_id'])
    op.create_index('ix_engagement_content', 'student_engagement_metrics', ['content_id'])
    op.create_index('ix_engagement_teacher_period', 'student_engagement_metrics', ['teacher_id', 'period_start', 'period_end'])
    
    # Learning outcome metrics
    op.create_table(
        'learning_outcome_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('teacher_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('total_assessments', sa.Integer, default=0),
        sa.Column('passed_count', sa.Integer, default=0),
        sa.Column('failed_count', sa.Integer, default=0),
        sa.Column('average_score', sa.Float, default=0.0),
        sa.Column('median_score', sa.Float, default=0.0),
        sa.Column('std_deviation', sa.Float, default=0.0),
        sa.Column('pre_assessment_avg', sa.Float, default=0.0),
        sa.Column('post_assessment_avg', sa.Float, default=0.0),
        sa.Column('improvement_percentage', sa.Float, default=0.0),
        sa.Column('knowledge_retention_rate', sa.Float, default=0.0),
        sa.Column('concept_mastery_rate', sa.Float, default=0.0),
        sa.Column('outcome_score', sa.Float, nullable=False),
        sa.Column('period_start', sa.TIMESTAMP, nullable=False),
        sa.Column('period_end', sa.TIMESTAMP, nullable=False),
        sa.Column('calculated_at', sa.TIMESTAMP, server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['teacher_id'], ['teacher_profiles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['content_id'], ['processed_content.id'], ondelete='CASCADE')
    )
    op.create_index('ix_outcome_teacher', 'learning_outcome_metrics', ['teacher_id'])
    op.create_index('ix_outcome_content', 'learning_outcome_metrics', ['content_id'])
    op.create_index('ix_outcome_teacher_period', 'learning_outcome_metrics', ['teacher_id', 'period_start', 'period_end'])
    
    # Teacher evaluations
    op.create_table(
        'teacher_evaluations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('teacher_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('evaluation_type', sa.String(50), nullable=False),
        sa.Column('period_start', sa.TIMESTAMP, nullable=False),
        sa.Column('period_end', sa.TIMESTAMP, nullable=False),
        sa.Column('content_quality_score', sa.Float, nullable=False),
        sa.Column('student_engagement_score', sa.Float, nullable=False),
        sa.Column('learning_outcome_score', sa.Float, nullable=False),
        sa.Column('overall_score', sa.Float, nullable=False),
        sa.Column('overall_rating', sa.Float, nullable=False),
        sa.Column('total_content_created', sa.Integer, default=0),
        sa.Column('total_students_taught', sa.Integer, default=0),
        sa.Column('average_completion_rate', sa.Float, default=0.0),
        sa.Column('average_student_satisfaction', sa.Float, default=0.0),
        sa.Column('strengths', postgresql.JSONB, default=[]),
        sa.Column('improvements_needed', postgresql.JSONB, default=[]),
        sa.Column('recommendations', postgresql.JSONB, default=[]),
        sa.Column('status', sa.String(50), default='draft'),
        sa.Column('evaluator_id', postgresql.UUID(as_uuid=True)),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('now()'), nullable=False),
        sa.Column('finalized_at', sa.TIMESTAMP),
        sa.ForeignKeyConstraint(['teacher_id'], ['teacher_profiles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['evaluator_id'], ['users.id'])
    )
    op.create_index('ix_eval_teacher', 'teacher_evaluations', ['teacher_id'])
    op.create_index('ix_eval_score', 'teacher_evaluations', ['overall_score'])
    op.create_index('ix_eval_teacher_period', 'teacher_evaluations', ['teacher_id', 'period_start', 'period_end'])


def downgrade() -> None:
    """Revert schema changes."""
    
    # Drop teacher evaluation tables
    op.drop_table('teacher_evaluations')
    op.drop_table('learning_outcome_metrics')
    op.drop_table('student_engagement_metrics')
    op.drop_table('content_quality_metrics')
    op.drop_table('teacher_profiles')
    
    # Drop token rotation tables
    op.drop_table('refresh_tokens')
    op.drop_table('token_blacklist')
