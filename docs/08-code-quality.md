# Code Quality

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## Quality Standards

The codebase adheres to strict quality standards across multiple dimensions:

| Dimension | Tool | Configuration |
|-----------|------|---------------|
| **Formatting** | Black, Prettier | Line length 100, double quotes |
| **Linting** | Ruff, ESLint | Strict mode with type checking |
| **Type Checking** | mypy, TypeScript | Strict mode |
| **Security** | Bandit, npm audit | High severity blocking |
| **Testing** | pytest, Jest | 80% coverage target |

---

## Python Code Quality

### Formatting with Black

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py311']
include = '\.pyi?$'
exclude = '''
/(
    \.git
    | \.venv
    | __pycache__
    | alembic/versions
)/
'''
```

### Linting with Ruff

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py311"
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # Pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "ARG",  # flake8-unused-arguments
    "SIM",  # flake8-simplify
]
ignore = [
    "E501",  # Line too long (handled by black)
    "B008",  # Function call in argument defaults
]

[tool.ruff.per-file-ignores]
"tests/*" = ["ARG"]  # Allow unused arguments in tests
"alembic/*" = ["E501"]  # Allow long lines in migrations
```

### Type Checking with mypy

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[[tool.mypy.overrides]]
module = [
    "transformers.*",
    "torch.*",
    "sentence_transformers.*",
]
ignore_missing_imports = true
```

### Security Scanning with Bandit

```toml
# pyproject.toml
[tool.bandit]
exclude_dirs = ["tests", "alembic"]
skips = ["B101"]  # Skip assert usage warning
```

**Recent Bandit Report:**
```json
{
  "generated_at": "2025-12-05T10:30:00Z",
  "metrics": {
    "files_scanned": 145,
    "issues_by_severity": {
      "UNDEFINED": 0,
      "LOW": 3,
      "MEDIUM": 0,
      "HIGH": 0
    }
  }
}
```

---

## TypeScript Code Quality

### ESLint Configuration

```javascript
// .eslintrc.js
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/strict-type-checked',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'prettier',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: ['./tsconfig.json'],
  },
  plugins: ['@typescript-eslint', 'react-refresh'],
  rules: {
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    '@typescript-eslint/explicit-function-return-type': 'warn',
    'react-refresh/only-export-components': 'warn',
    'react/react-in-jsx-scope': 'off',
  },
};
```

### Prettier Configuration

```json
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100
}
```

### TypeScript Configuration

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "jsx": "react-jsx"
  }
}
```

---

## Testing

### pytest Configuration

```toml
# pyproject.toml — [tool.pytest.ini_options]
testpaths = ["tests"]
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --cov=backend
    --cov-report=term-missing
    --cov-report=html
    -v
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
```

### Test Structure

```
tests/
├── unit/               # Isolated unit tests
│   ├── test_rag.py
│   ├── test_translation.py
│   └── test_pipeline.py
├── integration/        # Component integration tests
│   ├── test_api.py
│   └── test_database.py
├── e2e/               # End-to-end tests
│   └── test_full_flow.py
├── fixtures/          # Test fixtures
│   └── sample_data.py
└── conftest.py        # Shared fixtures
```

### Example Unit Test

```python
# tests/unit/test_rag.py
import pytest
import numpy as np
from backend.services.rag import BGEM3Embedder, BGEReranker

class TestBGEM3Embedder:
    @pytest.fixture
    def embedder(self):
        return BGEM3Embedder()

    def test_encode_single_text(self, embedder):
        texts = ["What is photosynthesis?"]
        embeddings = embedder.encode(texts)

        assert embeddings.shape == (1, 1024)
        assert np.linalg.norm(embeddings[0]) > 0

    def test_encode_batch(self, embedder):
        texts = ["Question 1", "Question 2", "Question 3"]
        embeddings = embedder.encode(texts)

        assert embeddings.shape == (3, 1024)

    def test_query_encoding_adds_prefix(self, embedder):
        query = "What is gravity?"
        embedding = embedder.encode_query(query)

        assert embedding.shape == (1024,)


class TestBGEReranker:
    @pytest.fixture
    def reranker(self):
        return BGEReranker()

    def test_rerank_returns_sorted_indices(self, reranker):
        query = "What is Newton's First Law?"
        documents = [
            "The cell is the basic unit of life.",
            "Newton's First Law states that an object at rest stays at rest.",
            "Photosynthesis converts light energy to chemical energy.",
        ]

        results = reranker.rerank(query, documents, top_k=2)

        assert len(results) == 2
        assert results[0][0] == 1  # Newton's law document should be first
        assert results[0][1] > results[1][1]  # Scores should be descending
```

### Example Integration Test

```python
# tests/integration/test_api.py
import pytest
from httpx import AsyncClient
from backend.api.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/api/v2/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_qa_endpoint_requires_auth(client):
    response = await client.post(
        "/api/v2/qa/ask",
        json={"question": "What is gravity?"}
    )

    assert response.status_code == 401

@pytest.mark.asyncio
async def test_qa_endpoint_with_auth(client, auth_token):
    response = await client.post(
        "/api/v2/qa/ask",
        json={"question": "What is gravity?", "stream": False},
        headers={"Authorization": f"Bearer {auth_token}"}
    )

    assert response.status_code == 200
    assert "answer" in response.json()
```

### Coverage Report

```
Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
backend/api/main.py                       245     18    93%
backend/services/rag.py                   312     25    92%
backend/services/translate/service.py     89      8    91%
backend/core/config.py                    156      0   100%
-----------------------------------------------------------
TOTAL                                    2847    284    90%
```

---

## Pre-commit Hooks

### Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
        args: ['--line-length=100']

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: ['--fix']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-redis
          - types-requests

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.7
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']
```

### Installation

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Quality Check Script

```bash
#!/bin/bash
# Example: scripts/testing/quality_checks.sh

set -e

echo "Running quality checks..."

echo "1. Formatting (Black)"
black --check backend/ tests/

echo "2. Linting (Ruff)"
ruff check backend/ tests/

echo "3. Type checking (mypy)"
mypy backend/

echo "4. Security scan (Bandit)"
bandit -r backend/ -c pyproject.toml

echo "5. Tests (pytest)"
pytest tests/ --cov=backend --cov-fail-under=80

echo "6. Frontend linting (ESLint)"
cd frontend && npm run lint

echo "7. Frontend type check (TypeScript)"
cd frontend && npm run type-check

echo "All quality checks passed!"
```

---

## Documentation Standards

### Docstring Format

```python
def encode(
    self,
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode texts to embeddings with GPU coordination.

    Args:
        texts: List of texts to encode
        batch_size: Batch size for encoding (default: 32)
        show_progress: Show progress bar (default: False)

    Returns:
        numpy array of embeddings with shape (len(texts), 1024)

    Raises:
        MemoryError: If insufficient GPU memory
        CircuitBreakerError: If ML service circuit is open

    Example:
        >>> embedder = BGEM3Embedder()
        >>> embeddings = embedder.encode(["Hello world"])
        >>> embeddings.shape
        (1, 1024)
    """
```

### Comment Guidelines

```python
# Good: Explains WHY
# Use sentence-transformers on Apple Silicon to avoid MPS watermark bugs
# in FlagEmbedding. This provides equivalent functionality with better
# memory stability on M-series chips.

# Bad: Explains WHAT (code already shows this)
# Load the model
self._model = SentenceTransformer(...)
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/quality.yml
name: Quality Checks

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run quality checks
        run: scripts/testing/quality_checks.sh

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

*For future improvements, see Section 10: Future Improvements.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com
