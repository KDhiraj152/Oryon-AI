"""Add user tracking to processed_content and feedback

Revision ID: 004_add_user_tracking
Revises: 61631d311ed9
Create Date: 2025-11-16 09:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = '004_add_user_tracking'
down_revision = '61631d311ed9'
branch_labels = None
depends_on = None


def upgrade():
    """Add user_id columns to processed_content and feedback tables."""
    from sqlalchemy import inspect
    from alembic import op
    
    conn = op.get_bind()
    inspector = inspect(conn)
    
    # Add user_id to processed_content if it doesn't exist
    columns = [col['name'] for col in inspector.get_columns('processed_content')]
    if 'user_id' not in columns:
        op.add_column(
            'processed_content',
            sa.Column('user_id', UUID(as_uuid=True), nullable=True)
        )
        op.create_index('ix_processed_content_user_id', 'processed_content', ['user_id'])
    
    # Add user_id to feedback if it doesn't exist
    columns = [col['name'] for col in inspector.get_columns('feedback')]
    if 'user_id' not in columns:
        op.add_column(
            'feedback',
            sa.Column('user_id', UUID(as_uuid=True), nullable=True)
        )
        op.create_index('ix_feedback_user_id', 'feedback', ['user_id'])


def downgrade():
    """Remove user_id columns."""
    op.drop_index('ix_feedback_user_id', 'feedback')
    op.drop_column('feedback', 'user_id')
    
    op.drop_index('ix_processed_content_user_id', 'processed_content')
    op.drop_column('processed_content', 'user_id')
