"""
Database Migration: A/B Testing Framework

Revision ID: 009_add_ab_testing
Revises: 008_add_q_a_tables
Create Date: 2025-01-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '009_add_ab_testing'
down_revision = '008_add_q_a_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create experiments table
    op.create_table(
        'experiments',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(), nullable=False, server_default='draft'),
        sa.Column('targeting', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_experiments_status', 'experiments', ['status'])
    op.create_index('idx_experiments_dates', 'experiments', ['start_date', 'end_date'])
    
    # Create experiment_variants table
    op.create_table(
        'experiment_variants',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('experiment_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('traffic_allocation', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_control', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE')
    )
    op.create_index('idx_experiment_variants_experiment_id', 'experiment_variants', ['experiment_id'])
    
    # Create experiment_assignments table
    op.create_table(
        'experiment_assignments',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('experiment_id', sa.String(), nullable=False),
        sa.Column('variant_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('assigned_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['variant_id'], ['experiment_variants.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('experiment_id', 'user_id', name='uq_experiment_user')
    )
    op.create_index('idx_assignments_experiment_user', 'experiment_assignments', ['experiment_id', 'user_id'])
    op.create_index('idx_assignments_variant', 'experiment_assignments', ['variant_id'])
    
    # Create experiment_events table
    op.create_table(
        'experiment_events',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('experiment_id', sa.String(), nullable=False),
        sa.Column('variant_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('event_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['variant_id'], ['experiment_variants.id'], ondelete='CASCADE')
    )
    op.create_index('idx_events_experiment_variant', 'experiment_events', ['experiment_id', 'variant_id'])
    op.create_index('idx_events_timestamp', 'experiment_events', ['timestamp'])
    op.create_index('idx_events_type', 'experiment_events', ['event_type'])


def downgrade() -> None:
    op.drop_table('experiment_events')
    op.drop_table('experiment_assignments')
    op.drop_table('experiment_variants')
    op.drop_table('experiments')
