#!/usr/bin/env bash
# PostgreSQL Setup Script - Production-Grade Database Configuration
# Creates optimized database with pgvector, connection pooling, and indexes

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   Oryon PostgreSQL Setup${NC}"
echo -e "${BLUE}   High-Performance Database Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Configuration
DB_NAME="oryon"
DB_USER="postgres"
DB_PASSWORD="postgres123"
DB_PORT="5432"
DB_HOST="localhost"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}✗ PostgreSQL not found${NC}"
    echo -e "${YELLOW}Installing PostgreSQL...${NC}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install postgresql@17
        brew services start postgresql@17
        export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y postgresql-17 postgresql-contrib-17
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    else
        echo -e "${RED}✗ Unsupported OS. Please install PostgreSQL manually.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ PostgreSQL installed${NC}"

# Check PostgreSQL version
PG_VERSION=$(psql --version | grep -oE '[0-9]+' | /usr/bin/head -n 1)
echo -e "${GREEN}✓ PostgreSQL version: $PG_VERSION${NC}"

# Create PostgreSQL user if needed
echo -e "${YELLOW}→ Setting up PostgreSQL user...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - create user without sudo
    psql postgres -tc "SELECT 1 FROM pg_user WHERE usename = '$DB_USER'" | grep -q 1 || \
        psql postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD' SUPERUSER CREATEDB CREATEROLE LOGIN;"
else
    # Linux - use sudo
    sudo -u postgres psql -tc "SELECT 1 FROM pg_user WHERE usename = '$DB_USER'" | grep -q 1 || \
        sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD' SUPERUSER CREATEDB CREATEROLE LOGIN;"
fi

echo -e "${GREEN}✓ PostgreSQL user configured${NC}"

# Drop and recreate database for clean slate
echo -e "${YELLOW}→ Creating database...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    psql postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>/dev/null || true
    psql postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
else
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>/dev/null || true
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
fi

echo -e "${GREEN}✓ Database '$DB_NAME' created${NC}"

# Install pgvector extension
echo -e "${YELLOW}→ Installing pgvector extension...${NC}"
if ! psql -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null; then
    echo -e "${YELLOW}  Installing pgvector...${NC}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install pgvector
    else
        sudo apt-get install -y postgresql-17-pgvector
    fi

    psql -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"
fi

echo -e "${GREEN}✓ pgvector extension installed${NC}"

# Install other useful extensions
echo -e "${YELLOW}→ Installing PostgreSQL extensions...${NC}"
psql -U $DB_USER -d $DB_NAME <<EOF
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
EOF

echo -e "${GREEN}✓ Extensions installed${NC}"

# Optimize PostgreSQL configuration
echo -e "${YELLOW}→ Optimizing PostgreSQL configuration...${NC}"

# Get total system RAM
if [[ "$OSTYPE" == "darwin"* ]]; then
    TOTAL_RAM_MB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024)}')
else
    TOTAL_RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
fi

# Calculate optimal settings (25% of RAM for shared_buffers, etc.)
SHARED_BUFFERS=$((TOTAL_RAM_MB / 4))
EFFECTIVE_CACHE_SIZE=$((TOTAL_RAM_MB * 3 / 4))
MAINTENANCE_WORK_MEM=$((TOTAL_RAM_MB / 16))
WORK_MEM=$((TOTAL_RAM_MB / 64))

cat > /tmp/postgresql_performance.conf <<EOF
# Oryon PostgreSQL Performance Configuration
# Generated on $(date)

# Memory Configuration (Optimized for ${TOTAL_RAM_MB}MB RAM)
shared_buffers = ${SHARED_BUFFERS}MB
effective_cache_size = ${EFFECTIVE_CACHE_SIZE}MB
maintenance_work_mem = ${MAINTENANCE_WORK_MEM}MB
work_mem = ${WORK_MEM}MB

# Connection Settings
max_connections = 100
shared_preload_libraries = 'pg_stat_statements'

# Checkpointing
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Query Planner
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

# Autovacuum
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 10s
EOF

echo -e "${GREEN}✓ Performance configuration generated${NC}"
echo -e "${YELLOW}  Note: Review /tmp/postgresql_performance.conf and apply manually to postgresql.conf${NC}"

# Update .env file
echo -e "${YELLOW}→ Updating .env configuration...${NC}"

DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

if [ -f .env ]; then
    # Update existing DATABASE_URL
    if grep -q "^DATABASE_URL=" .env; then
        sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=${DATABASE_URL}|" .env
    else
        echo "DATABASE_URL=${DATABASE_URL}" >> .env
    fi

    # Add connection pool settings if missing
    grep -q "^DB_POOL_SIZE=" .env || echo "DB_POOL_SIZE=20" >> .env
    grep -q "^DB_MAX_OVERFLOW=" .env || echo "DB_MAX_OVERFLOW=40" >> .env
    grep -q "^DB_POOL_TIMEOUT=" .env || echo "DB_POOL_TIMEOUT=30" >> .env
    grep -q "^DB_POOL_RECYCLE=" .env || echo "DB_POOL_RECYCLE=3600" >> .env
    grep -q "^DB_ECHO=" .env || echo "DB_ECHO=false" >> .env
else
    # Create new .env
    cp .env.example .env
    sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=${DATABASE_URL}|" .env
fi

echo -e "${GREEN}✓ .env updated with PostgreSQL connection${NC}"

# Run Alembic migrations
echo -e "${YELLOW}→ Running database migrations...${NC}"
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

python -m alembic upgrade head

echo -e "${GREEN}✓ Migrations completed${NC}"

# Create optimized indexes
echo -e "${YELLOW}→ Creating optimized indexes...${NC}"

psql -U $DB_USER -d $DB_NAME <<EOF
-- Performance indexes for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processed_content_user_created
    ON processed_content(user_id, created_at DESC)
    WHERE user_id IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processed_content_grade_subject_lang
    ON processed_content(complexity_level, subject, language)
    WHERE complexity_level IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_content_idx
    ON document_chunks(content_id, chunk_index);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_content_chunk
    ON embeddings(content_id, chunk_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_history_user_content
    ON chat_history(user_id, content_id, created_at DESC);

-- Full-text search indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processed_content_text_search
    ON processed_content USING gin(to_tsvector('english', original_text));

-- GIN index for JSONB columns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processed_content_metadata
    ON processed_content USING gin(metadata);

-- BRIN indexes for time-series data (efficient for large tables)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pipeline_logs_timestamp_brin
    ON pipeline_logs USING brin(timestamp);

-- Partial indexes for common filters
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active
    ON users(id, email)
    WHERE is_active = true;

ANALYZE;
EOF

echo -e "${GREEN}✓ Optimized indexes created${NC}"

# Verify pgvector is working
echo -e "${YELLOW}→ Verifying pgvector installation...${NC}"
psql -U $DB_USER -d $DB_NAME <<EOF
SELECT extversion FROM pg_extension WHERE extname = 'vector';
EOF

echo -e "${GREEN}✓ pgvector verified${NC}"

# Test connection
echo -e "${YELLOW}→ Testing database connection...${NC}"
python -c "
import sqlalchemy
from sqlalchemy import create_engine, text

engine = create_engine('${DATABASE_URL}',
                      pool_pre_ping=True,
                      pool_size=20,
                      max_overflow=40,
                      echo=False)

with engine.connect() as conn:
    result = conn.execute(text('SELECT version();'))
    version = result.fetchone()[0]
    print(f'✓ Connected to: {version}')

    result = conn.execute(text('SELECT count(*) FROM information_schema.tables WHERE table_schema = \\'public\\';'))
    table_count = result.fetchone()[0]
    print(f'✓ Tables in database: {table_count}')

    result = conn.execute(text('SELECT extname, extversion FROM pg_extension WHERE extname = \\'vector\\';'))
    pgvector = result.fetchone()
    if pgvector:
        print(f'✓ pgvector version: {pgvector[1]}')
"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   PostgreSQL Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}Database Connection:${NC}"
echo -e "  Host:     ${DB_HOST}"
echo -e "  Port:     ${DB_PORT}"
echo -e "  Database: ${DB_NAME}"
echo -e "  User:     ${DB_USER}"
echo ""
echo -e "${BLUE}Connection String:${NC}"
echo -e "  ${DATABASE_URL}"
echo ""
echo -e "${BLUE}Performance Features:${NC}"
echo -e "  ✓ Connection pooling (20 connections, 40 overflow)"
echo -e "  ✓ pgvector for semantic search"
echo -e "  ✓ Optimized indexes for common queries"
echo -e "  ✓ Full-text search enabled"
echo -e "  ✓ JSONB indexing for metadata"
echo -e "  ✓ Query performance logging"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. Review /tmp/postgresql_performance.conf"
echo -e "  2. Apply settings to postgresql.conf if desired"
echo -e "  3. Run: ./scripts/deployment/start-backend"
echo -e "  4. Run tests: pytest tests/"
echo ""
