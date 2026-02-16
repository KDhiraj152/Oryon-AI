"""Add feedback table

Revision ID: 002_add_feedback
Revises: 001_initial_schema
Create Date: 2024-12-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = '002_add_feedback'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def upgrade():
    """Create feedback table."""
    op.create_table(
        'feedback',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('content_id', UUID(as_uuid=True), sa.ForeignKey('processed_content.id'), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('feedback_text', sa.Text()),
        sa.Column('issue_type', sa.String(100)),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'))
    )
    
    # Create index on content_id for faster queries
    op.create_index('ix_feedback_content_id', 'feedback', ['content_id'])
    
    # Create index on rating for analytics
    op.create_index('ix_feedback_rating', 'feedback', ['rating'])


def downgrade():
    """Drop feedback table."""
    op.drop_index('ix_feedback_rating')
    op.drop_index('ix_feedback_content_id')
    op.drop_table('feedback')
