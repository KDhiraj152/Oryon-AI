#!/usr/bin/env python3
"""Verify restructuring integrity."""
import sys
sys.path.insert(0, ".")

def main():
    passed = 0
    failed = 0

    # Test 1: core/types.py imports
    try:
        from backend.core.types import ModelTier, TaskType, ModelType
        print("✓ core/types.py imports OK")
        passed += 1
    except Exception as e:
        print(f"✗ core/types.py: {e}")
        failed += 1

    # Test 2: model_config uses canonical types
    try:
        from backend.core.types import ModelTier, TaskType
        from backend.core.model_config import ModelTier as MT, TaskType as TT
        assert MT is ModelTier, "ModelTier not canonical"
        assert TT is TaskType, "TaskType not canonical"
        print("✓ model_config.py uses canonical types")
        passed += 1
    except Exception as e:
        print(f"✗ model_config canonical: {e}")
        failed += 1

    # Test 3: api/deps.py imports
    try:
        from backend.api.deps import get_ai_engine, get_pipeline, json_dumps
        print("✓ api/deps.py imports OK")
        passed += 1
    except Exception as e:
        print(f"✗ api/deps.py: {e}")
        failed += 1

    # Test 4: AgentRegistry singleton
    try:
        from backend.agents.base import AgentRegistry
        r1 = AgentRegistry()
        r2 = AgentRegistry()
        assert r1 is r2, "AgentRegistry not singleton"
        AgentRegistry.reset()
        print("✓ AgentRegistry singleton works")
        passed += 1
    except Exception as e:
        print(f"✗ AgentRegistry: {e}")
        failed += 1

    # Test 5: Backward aliases
    try:
        from backend.core.optimized.model_manager import ModelConfig, HardwareModelConfig
        assert ModelConfig is HardwareModelConfig, "alias broken"
        print("✓ HardwareModelConfig alias OK")
        passed += 1
    except Exception as e:
        print(f"✗ HardwareModelConfig alias: {e}")
        failed += 1

    try:
        from backend.core.optimized.device_router import TaskType as DT, DeviceTaskType
        assert DT is DeviceTaskType, "alias broken"
        print("✓ DeviceTaskType alias OK")
        passed += 1
    except Exception as e:
        print(f"✗ DeviceTaskType alias: {e}")
        failed += 1

    # Test 6: GPU coordination
    try:
        from backend.services.inference.gpu_coordination import (
            get_gpu_lock, get_mlx_lock, get_gpu_semaphore, get_mlx_semaphore,
            run_on_gpu_sync, MPS_SEMAPHORE_LIMIT, MLX_SEMAPHORE_LIMIT
        )
        print("✓ gpu_coordination.py imports OK")
        passed += 1
    except Exception as e:
        print(f"✗ gpu_coordination: {e}")
        failed += 1

    # Test 7: Inference re-exports
    try:
        from backend.services.inference.gpu_coordination import get_gpu_lock, run_on_gpu_sync
        from backend.services.inference import get_gpu_lock as ggl, run_on_gpu_sync as rgs
        assert ggl is get_gpu_lock
        assert rgs is run_on_gpu_sync
        print("✓ inference/__init__.py re-exports OK")
        passed += 1
    except Exception as e:
        print(f"✗ inference re-exports: {e}")
        failed += 1

    # Test 8: CircuitBreaker consolidation
    try:
        from backend.core.circuit_breaker import CircuitBreaker as CB1
        from backend.core.exceptions import CircuitBreaker as CB2
        assert CB1 is CB2, "not consolidated"
        print("✓ CircuitBreaker consolidated OK")
        passed += 1
    except Exception as e:
        print(f"✗ CircuitBreaker: {e}")
        failed += 1

    # Test 9: json_dumps
    try:
        from backend.api.deps import json_dumps
        result = json_dumps({"test": True})
        # orjson returns bytes, stdlib json returns str
        assert "test" in (result if isinstance(result, str) else result.decode())
        print("✓ json_dumps works")
        passed += 1
    except Exception as e:
        print(f"✗ json_dumps: {e}")
        failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All verification checks passed ✓")
    sys.exit(failed)

if __name__ == "__main__":
    main()
