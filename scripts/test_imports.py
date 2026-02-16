#!/usr/bin/env python3
"""Test that imports work after reorganization."""
import sys

sys.path.insert(0, '.')

tests = [
    ('backend.infra.cache.multi_tier_cache', 'get_unified_cache'),
    ('backend.infra.hardware.device', 'DeviceManager'),
    ('backend.infra.telemetry.metrics', 'MetricsCollector'),
    ('backend.db.database', 'get_db'),
    ('backend.ml.pipeline.unified_pipeline', 'UnifiedPipelineService'),
    ('backend.services.chat.engine', 'AIEngine'),
    ('backend.api.middleware.orchestrator', 'MiddlewareOrchestrator'),
    ('backend.core.config', 'get_settings'),
    ('backend.api.main', 'app'),
]

passed = 0
failed = 0

for module, attr in tests:
    try:
        mod = __import__(module, fromlist=[attr])
        getattr(mod, attr)
        print(f'OK  {module}.{attr}')
        passed += 1
    except Exception as e:
        print(f'ERR {module}.{attr}: {e}')
        failed += 1

print(f'\nPassed: {passed}, Failed: {failed}')
sys.exit(0 if failed == 0 else 1)
