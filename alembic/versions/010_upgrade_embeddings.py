"""Upgrade embeddings to E5-large (1024-dim)

Revision ID: 009_upgrade_embeddings
Revises: 008
Create Date: 2025-11-28 20:00:00.000000

This migration upgrades the embedding system from sentence-transformers/all-MiniLM-L6-v2 (384-dim)
to intfloat/multilingual-e5-large (1024-dim) for better multilingual semantic search.

IMPORTANT: This migration does NOT automatically re-embed existing data. 
After running this migration, you should run a background task to re-generate embeddings
for existing documents using the new E5-large model.

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '010_upgrade_embeddings'
down_revision = '009_add_ab_testing'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade to E5-large embeddings."""
    
    # Add embedding_version column to track which model generated the embedding
    op.add_column('embeddings', 
        sa.Column('embedding_version', sa.Integer(), nullable=True, server_default='1')
    )
    
    # Update default embedding model to E5-large
    op.execute("""
        UPDATE embeddings 
        SET embedding_model = 'intfloat/multilingual-e5-large',
            embedding_version = 2
        WHERE embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
    """)
    
    # Note: The actual vector column dimension change requires recreating the column
    # This is commented out because it would lose existing embeddings
    # Instead, we'll handle dimension mismatch in the application code
    
    # To fully migrate, you would need to:
    # 1. Create a new column with 1024 dimensions
    # 2. Re-generate all embeddings with E5-large
    # 3. Drop the old column
    # 4. Rename the new column
    
    # Example (DO NOT RUN without re-embedding data):
    # op.add_column('embeddings', sa.Column('embedding_v2', postgresql.ARRAY(sa.Float()), nullable=True))
    # # Run background task to re-embed all data
    # op.drop_column('embeddings', 'embedding')
    # op.alter_column('embeddings', 'embedding_v2', new_column_name='embedding')
    
    print("""
    ========================================
    MIGRATION COMPLETED
    ========================================
    
    NEXT STEPS:
    1. Run the re-embedding script to convert existing data:
       python scripts/re_embed_documents.py
    
    2. The script will:
       - Load each document chunk
       - Generate new E5-large embeddings (1024-dim)
       - Update the embeddings table
       - Mark as embedding_version=2
    
    3. Until re-embedding is complete:
       - New embeddings will use E5-large (1024-dim)
       - Old embeddings will use MiniLM (384-dim)
       - The RAG service will handle dimension differences
    
    ========================================
    """)


def downgrade():
    """Downgrade to MiniLM embeddings."""
    
    # Revert to old embedding model
    op.execute("""
        UPDATE embeddings 
        SET embedding_model = 'sentence-transformers/all-MiniLM-L6-v2',
            embedding_version = 1
        WHERE embedding_model = 'intfloat/multilingual-e5-large'
    """)
    
    # Remove embedding_version column
    op.drop_column('embeddings', 'embedding_version')
    
    print("""
    ========================================
    DOWNGRADE COMPLETED
    ========================================
    
    Reverted to sentence-transformers/all-MiniLM-L6-v2 (384-dim)
    
    WARNING: If you had re-embedded data with E5-large,
    you will need to re-embed again with MiniLM.
    
    ========================================
    """)
