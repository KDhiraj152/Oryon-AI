#!/bin/bash
# Quick Demo Startup Script for ShikshaSetu
# This script starts all services needed for a working demo

set -e

echo "=========================================="
echo "ShikshaSetu Demo Startup"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    # Generate random passwords for development
    DEV_DB_PASS=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
    cat > .env << EOF
# Database - auto-generated development credentials
DATABASE_URL=postgresql://shiksha_user:${DEV_DB_PASS}@localhost:5432/shiksha_setu

# Redis & Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# JWT Secret (auto-generated)
JWT_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(64))')

# Environment
ENVIRONMENT=development
DEBUG=true
DEPLOYMENT_TIER=local

# Device & Models
DEVICE=auto
USE_QUANTIZATION=true
MODEL_CACHE_DIR=data/models

# API Keys (optional - for Bhashini integration)
BHASHINI_USER_ID=08f28ee8e9-113a-49a9-8e5c-cf1b07df5708
BHASHINI_API_KEY=IuC2opogcScY_Zz9uSV4ewMLuPleqtKalnwrpQGzgh6YNApgw_p6TGwB2RiNyg2B

# Upload limits
MAX_UPLOAD_SIZE=104857600
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
fi

# Check if services are running
echo ""
echo "Checking required services..."

# Check PostgreSQL
if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL is running${NC}"
else
    echo -e "${YELLOW}⚠ PostgreSQL not running. Starting with Docker...${NC}"
    docker run -d \
        --name shiksha-postgres \
        -e POSTGRES_USER=shiksha_user \
        -e POSTGRES_PASSWORD=shiksha_pass \
        -e POSTGRES_DB=shiksha_setu \
        -p 5432:5432 \
        pgvector/pgvector:pg16 || echo "PostgreSQL container already exists"
    sleep 3
fi

# Check Redis
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${YELLOW}⚠ Redis not running. Starting with Docker...${NC}"
    docker run -d \
        --name shiksha-redis \
        -p 6379:6379 \
        redis:7-alpine || echo "Redis container already exists"
    sleep 2
fi

# Run migrations
echo ""
echo "Running database migrations..."
export $(cat .env | grep -v '^#' | xargs)
alembic upgrade head
echo -e "${GREEN}✓ Migrations complete${NC}"

# Create demo user if not exists
echo ""
echo "Setting up demo user..."
python3 -c "
from backend.database import get_db_session
from backend.models import User
from backend.utils.auth import get_password_hash
import uuid

with get_db_session() as db:
    existing = db.query(User).filter(User.username == 'demo').first()
    if not existing:
        demo_user = User(
            id=uuid.uuid4(),
            username='demo',
            email='demo@shiksha.edu',
            hashed_password=get_password_hash('demo123'),
            full_name='Demo User',
            is_active=True
        )
        db.add(demo_user)
        db.commit()
        print('✓ Demo user created: username=demo, password=demo123')
    else:
        print('✓ Demo user already exists')
" || echo "Could not create demo user (will create on first startup)"

# Start Celery worker in background
echo ""
echo "Starting Celery worker..."
celery -A backend.tasks.celery_app worker --loglevel=info --pool=solo &
CELERY_PID=$!
echo -e "${GREEN}✓ Celery worker started (PID: $CELERY_PID)${NC}"

# Give Celery time to start
sleep 2

# Start FastAPI server
echo ""
echo "Starting FastAPI server..."
echo ""
echo "=========================================="
echo -e "${GREEN}Demo is ready!${NC}"
echo "=========================================="
echo ""
echo "API Server: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Login credentials:"
echo "  Username: demo"
echo "  Password: demo123"
echo ""
echo "Quick Test:"
echo "  1. Login at /api/v2/auth/login"
echo "  2. Upload document at /api/v2/content/simplify"
echo "  3. Ask questions at /api/v2/chat/guest"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=========================================="
echo ""

# Trap Ctrl+C and cleanup
trap 'echo ""; echo "Stopping services..."; kill $CELERY_PID 2>/dev/null; exit 0' INT

# Start uvicorn
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
