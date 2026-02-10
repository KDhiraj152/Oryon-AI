#!/usr/bin/env python3
"""Direct test to find blocking component in main.py lifespan."""

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_components():
    """Test each lifespan component individually."""

    # 1. Test Redis
    logger.info("Testing Redis connection...")
    try:
        import redis.asyncio as redis

        redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
        await redis_client.ping()
        logger.info("✓ Redis connection OK")
        await redis_client.close()
    except Exception as e:
        logger.error(f"✗ Redis failed: {e}")

    # 2. Test Database init
    logger.info("Testing Database init...")
    try:
        from backend.database import get_engine

        engine = get_engine()
        logger.info("✓ Database engine OK")
    except Exception as e:
        logger.error(f"✗ Database failed: {e}")

    # 3. Test Device Router
    logger.info("Testing Device Router...")
    try:
        from backend.core.hardware.config import HardwareConfig
        from backend.core.hardware.device_router import DeviceRouter

        config = HardwareConfig()
        device_router = DeviceRouter(config)
        device_router.start()
        logger.info(f"✓ Device Router started on {device_router.get_optimal_device()}")
        device_router.stop()
    except Exception as e:
        logger.error(f"✗ Device Router failed: {e}")

    # 4. Test Memory Coordinator
    logger.info("Testing Memory Coordinator...")
    try:
        from backend.core.memory.coordinator import MemoryCoordinator
        from backend.core.memory.pressure import MemoryPressure

        coordinator = MemoryCoordinator()

        def on_memory_pressure(pressure: MemoryPressure):
            logger.warning(f"Memory pressure: {pressure}")

        coordinator.register_pressure_callback(on_memory_pressure)
        logger.info("✓ Memory Coordinator created")

        # Test start_monitor - this might block!
        logger.info("  Starting monitor task...")
        task = coordinator.start_monitor(interval=60.0)  # Long interval
        logger.info(f"  ✓ Monitor started: {task}")

        # Give it a moment then cancel
        await asyncio.sleep(0.5)
        logger.info("  ✓ Monitor running OK (not blocking)")

    except Exception as e:
        logger.error(f"✗ Memory Coordinator failed: {e}")

    # 5. Test Model Discovery
    logger.info("Testing Model Discovery...")
    try:
        from backend.services.model_loader import discover_models

        await asyncio.wait_for(discover_models(), timeout=5.0)
        logger.info("✓ Model Discovery OK")
    except TimeoutError:
        logger.warning(
            "⚠ Model Discovery timed out after 5s (might be slow, not blocking)"
        )
    except Exception as e:
        logger.error(f"✗ Model Discovery failed: {e}")

    # 6. Test Policy Engine
    logger.info("Testing Policy Engine...")
    try:
        from backend.policy.engine import ContentPolicyEngine

        engine = ContentPolicyEngine()
        logger.info("✓ Policy Engine OK")
    except Exception as e:
        logger.error(f"✗ Policy Engine failed: {e}")

    # 7. Test Rate Limiter Middleware Init
    logger.info("Testing Rate Limiter creation...")
    try:
        from backend.core.optimized.rate_limiter import RateLimitMiddleware

        # Just test that it can be created
        logger.info("✓ RateLimitMiddleware importable")
    except Exception as e:
        logger.error(f"✗ Rate Limiter failed: {e}")

    # 8. Test Warmup (with timeout)
    logger.info("Testing Model Warmup...")
    try:
        from backend.api.main import _warm_up_models_async

        await asyncio.wait_for(_warm_up_models_async(), timeout=10.0)
        logger.info("✓ Model Warmup OK")
    except TimeoutError:
        logger.warning(
            "⚠ Model Warmup timed out after 10s (slow but not blocking if async)"
        )
    except Exception as e:
        logger.error(f"✗ Model Warmup failed: {e}")

    logger.info("=" * 50)
    logger.info("Component tests complete")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_components())
