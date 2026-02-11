# Scripts Directory

Organized utility scripts for development, testing, and deployment.

## Directory Structure

```text
scripts/
├── setup/           # Environment and dependency setup
├── deployment/      # Production deployment and operations
├── testing/         # Test suites and smoke tests
├── validation/      # System and configuration validation
├── benchmarks/      # Performance benchmarking
├── demo/            # Demonstrations and examples
└── utils/           # Developer utility scripts
```

## Naming Convention

All scripts use **snake_case** naming:

- Shell scripts: `script_name.sh`
- Python scripts: `script_name.py`

## Setup

```bash
# Full environment setup
bash scripts/setup/setup_python311.sh
python scripts/setup/setup_complete.py

# Download ML models
python scripts/setup/download_models.py
```

## Testing

```bash
# Quick smoke test
bash scripts/testing/smoke_quick.sh

# Full smoke test
bash scripts/testing/smoke_test.sh

# Quality checks (lint + security + tests)
bash scripts/testing/quality_checks.sh

# Run all tests via pytest
bash scripts/testing/test.sh
```

## Benchmarks

```bash
# API performance
python scripts/benchmarks/benchmark_api.py

# Hardware (SIMD, GPU, memory)
python scripts/benchmarks/benchmark_hardware.py

# Apple Silicon specific
bash scripts/benchmarks/benchmark_apple_silicon.sh

# Comprehensive
python scripts/benchmarks/benchmark_full.py
```

## Validation

```bash
# Startup validation
python scripts/validation/validate.py

# System components
bash scripts/validation/validate_system.sh

# Production readiness
bash scripts/validation/validate_production.sh
```
