# Content Pipeline Orchestrator

## Overview

The Content Pipeline Orchestrator coordinates the four-stage educational content processing pipeline:

1. **Simplification** (Flan-T5) - Adapts content complexity for grade levels 5-12
2. **Translation** (IndicTrans2) - Translates to Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi)
3. **Validation** (BERT) - Verifies NCERT alignment and semantic accuracy
4. **Speech Generation** (VITS/Bhashini) - Produces audio output

## Features

### Parameter Validation
- Validates all input parameters before processing
- Supports 5 Indian languages (MVP)
- Grade levels 5-12
- 6 subject areas
- 3 output formats (text, audio, both)

### Retry Logic
- Automatic retry with exponential backoff
- Up to 3 retry attempts per stage
- Backoff: 2^retry_count seconds

### Metrics Tracking
- Processing time per stage
- Success/failure status
- Error messages
- Retry counts
- Timestamps

### Quality Thresholds
- NCERT alignment: ≥80%
- Audio accuracy: ≥90% (ASR validation)

## Usage

```python
from src.pipeline.orchestrator import ContentPipelineOrchestrator

# Initialize orchestrator
orchestrator = ContentPipelineOrchestrator()

# Process content
result = orchestrator.process_content(
    input_data="Educational content about mathematics...",
    target_language="Hindi",
    grade_level=8,
    subject="Mathematics",
    output_format="both"
)

# Access results
print(f"Content ID: {result.id}")
print(f"NCERT Score: {result.ncert_alignment_score}")
print(f"Status: {result.validation_status}")
```

## Configuration

The orchestrator uses configuration from `config.py`:
- Database connection settings
- Model endpoints (Hugging Face)
- API keys
- Storage paths
- Rate limiting

## Error Handling

### PipelineValidationError
Raised when input parameters are invalid:
- Empty input data
- Unsupported language/subject/format
- Invalid grade level

### PipelineStageError
Raised when a stage fails after max retries:
- Model inference failures
- Network errors
- Quality threshold violations

## Testing

Run tests with:
```bash
python -m pytest tests/test_orchestrator_validation.py -v
```

## Example

See `examples/orchestrator_example.py` for complete usage examples.

## Requirements

- Python 3.9+
- PostgreSQL database
- Hugging Face API key
- Dependencies from requirements.txt
