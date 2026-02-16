"""Add learning recommendations

Revision ID: 015_add_learning_recommendations
Revises: 014_add_translation_review
Create Date: 2024-01-15 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '015_add_learning_recommendations'
down_revision = '014_add_translation_review'
branch_labels = None
depends_on = None


def upgrade():
    # User performance tracking
    op.create_table(
        'user_performance',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_type', sa.String(50), nullable=False),
        sa.Column('interaction_type', sa.String(50), nullable=False),
        sa.Column('score', sa.Float, nullable=True),
        sa.Column('time_spent', sa.Integer, nullable=True),
        sa.Column('completed', sa.Boolean, default=False),
        sa.Column('difficulty_rating', sa.SmallInteger, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    
    op.create_index('idx_user_performance_user_id', 'user_performance', ['user_id'])
    op.create_index('idx_user_performance_content_id', 'user_performance', ['content_id'])
    op.create_index('idx_user_performance_created_at', 'user_performance', ['created_at'])
    
    # Recommended content
    op.create_table(
        'recommended_content',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content_type', sa.String(50), nullable=False),
        sa.Column('recommendation_score', sa.Float, nullable=False),
        sa.Column('recommendation_reason', sa.Text, nullable=True),
        sa.Column('algorithm', sa.String(50), nullable=False),
        sa.Column('is_clicked', sa.Boolean, default=False),
        sa.Column('is_completed', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        
        sa.UniqueConstraint('user_id', 'content_id', name='uq_user_content_recommendation')
    )
    
    op.create_index('idx_recommended_content_user_id', 'recommended_content', ['user_id'])
    op.create_index('idx_recommended_content_score', 'recommended_content', ['recommendation_score'])
    
    # Learning paths
    op.create_table(
        'learning_paths',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('path_name', sa.String(255), nullable=False),
        sa.Column('goal', sa.Text, nullable=True),
        sa.Column('current_step', sa.Integer, default=0),
        sa.Column('total_steps', sa.Integer, nullable=False),
        sa.Column('progress_percentage', sa.Float, default=0.0),
        sa.Column('estimated_completion_hours', sa.Float, nullable=True),
        sa.Column('difficulty_level', sa.SmallInteger, default=1),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    
    op.create_index('idx_learning_paths_user_id', 'learning_paths', ['user_id'])
    op.create_index('idx_learning_paths_is_active', 'learning_paths', ['is_active'])


def downgrade():
    op.drop_table('learning_paths')
    op.drop_table('recommended_content')
    op.drop_table('user_performance')
