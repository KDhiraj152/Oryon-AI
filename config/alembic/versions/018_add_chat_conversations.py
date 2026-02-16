"""Add conversation and message tables for chat interface

Revision ID: 018_add_chat_conversations
Revises: 017_normalize_schema_fix_fk
Create Date: 2025-01-24 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP


# revision identifiers, used by Alembic.
revision = '018_add_chat_conversations'
down_revision = '017_normalize_schema_fix_fk'
branch_labels = None
depends_on = None

# Constants for server defaults
GEN_RANDOM_UUID = 'gen_random_uuid()'


def upgrade():
    """Create conversations, messages, and file_uploads tables."""

    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text(GEN_RANDOM_UUID)),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(255), server_default='New Chat'),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('metadata', JSONB, server_default='{}')
    )

    # Create index on user_id for faster lookups
    op.create_index('ix_conversations_user_id', 'conversations', ['user_id'])
    op.create_index('ix_conversations_updated_at', 'conversations', ['updated_at'])

    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text(GEN_RANDOM_UUID)),
        sa.Column('conversation_id', UUID(as_uuid=True), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(20), server_default='user'),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('metadata', JSONB, server_default='{}')
    )

    # Create index on conversation_id for faster message retrieval
    op.create_index('ix_messages_conversation_id', 'messages', ['conversation_id'])
    op.create_index('ix_messages_created_at', 'messages', ['created_at'])

    # Create file_uploads table
    op.create_table(
        'file_uploads',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text(GEN_RANDOM_UUID)),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('filename', sa.String(500), nullable=False),
        sa.Column('original_filename', sa.String(500), nullable=False),
        sa.Column('content_type', sa.String(100)),
        sa.Column('size', sa.String(50)),
        sa.Column('file_size', sa.String(50)),
        sa.Column('path', sa.String(1000)),
        sa.Column('url', sa.String(1000)),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('metadata', JSONB, server_default='{}')
    )

    # Create index on user_id for file_uploads
    op.create_index('ix_file_uploads_user_id', 'file_uploads', ['user_id'])


def downgrade():
    """Drop chat-related tables."""
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('file_uploads')
