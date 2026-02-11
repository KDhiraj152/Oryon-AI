# Deployment

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## Deployment Options

Shiksha Setu supports multiple deployment configurations:

| Mode | Use Case | Hardware | Complexity |
|------|----------|----------|------------|
| **Development** | Local testing | Any Mac/PC | Simple |
| **Single-Node Production** | Small school/organization | M4 Pro or RTX 4060+ | Moderate |
| **Multi-Node Production** | Large institution | Multiple GPUs | Advanced |

---

## Quick Start (Development)

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 17 with pgvector extension
- Redis 7+
- ~30GB disk space (models + data)

### Setup Commands

```bash
# Clone repository
git clone https://github.com/kdhiraj/shiksha-setu.git
cd shiksha-setu

# Run setup script
./setup.sh

# This script:
# 1. Creates Python virtual environment
# 2. Installs Python dependencies from requirements.txt
# 3. Installs Node.js dependencies (frontend)
# 4. Downloads AI models (~10GB)
# 5. Initializes PostgreSQL with pgvector
# 6. Runs Alembic migrations
# 7. Creates default configuration

# Start all services
./start.sh

# Access points:
# Frontend: http://localhost:5173
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Environment Configuration

Create `.env` file in project root:

```bash
# Application
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://shiksha:password@localhost:5432/shiksha_setu

# Redis
REDIS_URL=redis://localhost:6379/0

# Device (auto | cuda | mps | cpu)
DEVICE=auto

# Model Configuration
USE_QUANTIZATION=true
QUANTIZATION_TYPE=int4
MAX_GPU_MEMORY_GB=12

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

---

## Docker Deployment

### Docker Compose (Recommended)

The `docker-compose.yml` provides a complete production stack:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://shiksha:password@postgres:5432/shiksha_setu
      - REDIS_URL=redis://redis:6379/0
      - DEVICE=auto
    volumes:
      - ./storage:/app/storage
      - ./data/models:/app/data/models
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  postgres:
    image: pgvector/pgvector:pg17
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: shiksha
      POSTGRES_PASSWORD: password
      POSTGRES_DB: shiksha_setu
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./infrastructure/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./infrastructure/nginx/ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend

volumes:
  postgres_data:
  redis_data:
```

### Build and Run

```bash
# Build all containers
docker compose build

# Start services
docker compose up -d

# View logs
docker compose logs -f backend

# Stop services
docker compose down
```

### Backend Dockerfile

```dockerfile
# infrastructure/docker/Dockerfile.backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ ./backend/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create storage directories
RUN mkdir -p storage/audio storage/uploads storage/cache data/models

# Run with uvicorn
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Nginx Configuration

### Production Configuration

```nginx
# infrastructure/nginx/nginx.conf
upstream backend {
    server backend:8000;
}

upstream frontend {
    server frontend:3000;
}

server {
    listen 80;
    server_name shiksha-setu.local;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name shiksha-setu.local;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # API
    location /api/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
    }

    # Health check
    location /health {
        proxy_pass http://backend/api/v2/health;
    }

    # Static files caching
    location /static/ {
        alias /app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Gzip compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;
    gzip_min_length 1000;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    location /api/v2/qa/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://backend;
    }
}
```

---

## Monitoring Stack

### Prometheus Configuration

```yaml
# infrastructure/monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'shiksha-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: /api/v2/metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboard

Key metrics to monitor:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `request_latency_seconds` | API response time | p95 > 3s |
| `model_inference_seconds` | LLM generation time | p95 > 5s |
| `memory_usage_bytes` | RAM consumption | > 14GB |
| `gpu_utilization` | GPU usage percentage | < 20% (underutilized) |
| `active_requests` | Concurrent requests | > 50 |
| `error_rate` | 5xx responses | > 1% |

---

## Startup Scripts

### start.sh

```bash
#!/bin/bash
set -e

echo "Starting Shiksha Setu..."

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js required"; exit 1; }

# Activate virtual environment
source venv/bin/activate

# Start Redis (if not running)
if ! pgrep -x "redis-server" > /dev/null; then
    redis-server --daemonize yes
fi

# Start PostgreSQL (if not running)
if ! pg_isready -q; then
    pg_ctl start -D /usr/local/var/postgres -l /tmp/postgres.log
fi

# Run migrations
alembic upgrade head

# Start backend
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend
cd frontend && npm run dev &
FRONTEND_PID=$!

echo "Backend running on http://localhost:8000"
echo "Frontend running on http://localhost:5173"
echo "API Docs: http://localhost:8000/docs"

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
```

### stop.sh

```bash
#!/bin/bash

echo "Stopping Shiksha Setu..."

# Kill backend
pkill -f "uvicorn backend.api.main:app" || true

# Kill frontend
pkill -f "vite" || true

echo "Services stopped"
```

---

## Database Migrations

### Alembic Configuration

```ini
# alembic.ini
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = postgresql://shiksha:password@localhost:5432/shiksha_setu

[loggers]
keys = root,sqlalchemy,alembic

[logger_alembic]
level = INFO
handlers =
qualname = alembic
```

### Running Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1

# View migration history
alembic history
```

---

## Health Checks

### Liveness Probe

```bash
curl http://localhost:8000/api/v2/health
# {"status": "healthy", "version": "4.0.0"}
```

### Readiness Probe

```bash
curl http://localhost:8000/api/v2/health/ready
# {"ready": true, "checks": {"database": "ok", "redis": "ok", "models": {...}}}
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /api/v2/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /api/v2/health/ready
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 5
```

---

## Performance Tuning

### Uvicorn Workers

```bash
# Development (single worker, auto-reload)
uvicorn backend.api.main:app --reload

# Production (multiple workers)
uvicorn backend.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### PostgreSQL Tuning

```sql
-- Increase shared buffers for vector operations
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET work_mem = '256MB';

-- Optimize for SSD
ALTER SYSTEM SET random_page_cost = 1.1;

-- Reload configuration
SELECT pg_reload_conf();
```

### Redis Tuning

```bash
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save ""  # Disable persistence for cache-only use
```

---

## Backup and Recovery

### Database Backup

```bash
# Full backup
pg_dump -Fc shiksha_setu > backup_$(date +%Y%m%d).dump

# Restore
pg_restore -d shiksha_setu backup_20251205.dump
```

### Model Backup

```bash
# Models are stored in data/models/
tar -czvf models_backup.tar.gz data/models/
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Too many models loaded | Reduce `MAX_GPU_MEMORY_GB` |
| MPS errors | Metal conflicts | Ensure single GPU operation at a time |
| Slow first request | Model loading | Enable model preloading in config |
| Database connection refused | PostgreSQL not running | `pg_ctl start` |
| Redis connection refused | Redis not running | `redis-server` |

### Log Locations

```
logs/
├── shiksha_setu.log    # Application logs
├── uvicorn.log         # Server logs
├── alembic.log         # Migration logs
└── error.log           # Error-only logs
```

---

*For code quality details, see Section 9: Code Quality.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com
