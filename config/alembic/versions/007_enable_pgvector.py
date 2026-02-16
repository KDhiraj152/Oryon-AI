"""
Alembic migration: Enable pgvector extension and create vector indexes

This migration:
1. Enables the pgvector extension
2. Ensures question_embedding column exists
3. Creates optimized IVFFLAT indexes for similarity search

Note: IVFFLAT indexes require sufficient data for clustering.
- Run with '-x concurrently=true' to create indexes non-blocking (CONCURRENTLY)
- Indexes are only created if table has >= 1000 rows
- For smaller tables, create indexes manually after data ingestion
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = '007_enable_pgvector'
down_revision = '005_add_composite_indexes'
branch_labels = None
depends_on = None


# Minimum row count for IVFFLAT index creation (clustering requires data)
MIN_ROWS_FOR_IVFFLAT = 1000


def _create_ivfflat_indexes_if_ready(table_name: str, column_name: str, min_rows: int = MIN_ROWS_FOR_IVFFLAT):
    """
    Create IVFFLAT indexes only if the table has sufficient rows.
    
    IVFFLAT indexes use clustering and produce poor results on small/empty tables.
    This function checks row count and optionally creates indexes CONCURRENTLY
    to avoid blocking the table.
    
    Args:
        table_name: Name of the table
        column_name: Name of the vector column
        min_rows: Minimum rows required before creating index (default: 1000)
    
    Usage:
        Run migration normally: alembic upgrade head
        Run with concurrent index creation: alembic upgrade head -x concurrently=true
    """
    context = op.get_context()
    bind = op.get_bind()
    
    # Check if table exists
    table_exists = bind.execute(text(f"""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        )
    """)).scalar()
    
    if not table_exists:
        print(f"⚠️  Table '{table_name}' does not exist. Skipping IVFFLAT index creation.")
        return
    
    # Check row count
    row_count = bind.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    
    if row_count < min_rows:
        print(f"⚠️  Table '{table_name}' has only {row_count} rows (minimum {min_rows} required).")
        print("   IVFFLAT indexes require sufficient data for clustering.")
        print("   Create indexes manually after data ingestion using:")
        print("   CREATE INDEX CONCURRENTLY questions_embedding_cosine_idx")
        print(f"   ON {table_name} USING ivfflat ({column_name} vector_cosine_ops) WITH (lists = 100);")
        return
    
    # Check if we should create indexes concurrently (non-blocking)
    use_concurrent = context.config.attributes.get('concurrently', 'false').lower() == 'true'
    
    # Determine number of lists based on row count
    # Rule of thumb: lists = rows / 1000, capped between 10 and 1000
    num_lists = max(10, min(1000, row_count // 1000))
    
    print(f"✓ Creating IVFFLAT indexes on '{table_name}.{column_name}' ({row_count} rows, {num_lists} lists)")
    
    # Index names
    cosine_idx = f"{table_name}_embedding_cosine_idx"
    
    if use_concurrent:
        print("  Using CONCURRENTLY mode (non-blocking)")
        
        # For CONCURRENTLY, we need AUTOCOMMIT mode (outside transaction)
        # Get a new connection with autocommit
        conn = bind.connect()
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        
        # Create cosine similarity index
        conn.execute(text(f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {cosine_idx}
            ON {table_name} USING ivfflat ({column_name} vector_cosine_ops)
            WITH (lists = {num_lists})
        """))
        
        # Optional: Create L2 distance index
        # Uncomment if your application uses L2 distance
        # conn.execute(text(f"""
        #     CREATE INDEX CONCURRENTLY IF NOT EXISTS {l2_idx}
        #     ON {table_name} USING ivfflat ({column_name} vector_l2_ops)
        #     WITH (lists = {num_lists})
        # """))
        
        conn.close()
    else:
        print("  Using standard CREATE INDEX (may block table)")
        
        # Standard index creation within transaction
        op.execute(f"""
            CREATE INDEX IF NOT EXISTS {cosine_idx}
            ON {table_name} USING ivfflat ({column_name} vector_cosine_ops)
            WITH (lists = {num_lists})
        """)
        
        # Optional: Create L2 distance index
        # Uncomment if your application uses L2 distance
        # l2_idx = f"{table_name}_embedding_l2_idx"
        # op.execute(f"""
        #     CREATE INDEX IF NOT EXISTS {l2_idx}
        #     ON {table_name} USING ivfflat ({column_name} vector_l2_ops)
        #     WITH (lists = {num_lists})
        # """)
    
    print("✓ IVFFLAT indexes created successfully")


def upgrade():
    # Enable pgvector extension (safe to run multiple times)
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Note: The questions table and question_embedding column do not exist in this schema.
    # This migration was originally designed for a different table structure.
    # The actual vector embeddings are stored in the 'embeddings' table created in 
    # migration 61631d311ed9.
    # 
    # Keeping this code commented for reference:
    # try:
    #     op.execute("""
    #         DO $$ 
    #         BEGIN
    #             IF NOT EXISTS (
    #                 SELECT 1 FROM information_schema.columns 
    #                 WHERE table_name = 'questions' 
    #                 AND column_name = 'question_embedding'
    #             ) THEN
    #                 ALTER TABLE questions 
    #                 ADD COLUMN question_embedding vector(384);
    #             END IF;
    #         END $$;
    #     """)
    # except Exception:
    #     pass  # Column already exists
    
    # Note: The questions table does not exist. Vector indexes are created on the
    # 'embeddings' table in migration 61631d311ed9 instead.
    # The code below demonstrates proper IVFFLAT index creation with concurrent support:
    
    # _create_ivfflat_indexes_if_ready(
    #     table_name='questions',
    #     column_name='question_embedding',
    #     min_rows=1000
    # )


def downgrade():
    # Note: Since we didn't create any indexes or columns in upgrade(),
    # there's nothing to drop here.
    # The vector extension is used by migration 61631d311ed9, so we keep it.
    pass
    
    # Original code kept for reference:
    # op.execute('DROP INDEX IF EXISTS questions_embedding_cosine_idx')
    # op.execute('DROP INDEX IF EXISTS questions_embedding_l2_idx')
    # op.execute('ALTER TABLE questions DROP COLUMN IF EXISTS question_embedding')
    # op.execute('DROP EXTENSION IF EXISTS vector CASCADE')
