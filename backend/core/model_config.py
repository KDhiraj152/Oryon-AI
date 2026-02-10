"""
Hot-Reloadable Model Configuration
==================================

Provides hot-reloadable model configuration from JSON file.
Changes to models.json are picked up without restart.
"""

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default config file path
CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = CONFIG_DIR / "models.json"


class ModelTier(str, Enum):
    """Model capability tiers."""

    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    STRONG = "strong"
    SPECIALIZED = "specialized"


class TaskType(str, Enum):
    """Task types for model routing."""

    CHAT = "chat"
    REASONING = "reasoning"
    CODE = "code"
    TRANSLATION = "translation"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    VALIDATION = "validation"
    SYSTEM = "system"


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_id: str
    tier: ModelTier
    max_tokens: int = 2048
    context_window: int = 4096
    avg_latency_ms: int = 100
    task_types: list[TaskType] | None = None

    def __post_init__(self):
        if self.task_types is None:
            self.task_types = [TaskType.CHAT]
        # Convert string task types to enum
        self.task_types = [
            TaskType(t) if isinstance(t, str) else t for t in self.task_types
        ]
        # Convert string tier to enum
        if isinstance(self.tier, str):
            self.tier = ModelTier(self.tier)


class ModelConfigLoader:
    """
    Hot-reloadable model configuration loader.

    Monitors the config file and reloads on changes.
    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        config_file: Path = DEFAULT_CONFIG_FILE,
        auto_reload: bool = True,
        reload_interval: float = 30.0,
    ):
        self._config_file = Path(config_file)
        self._auto_reload = auto_reload
        self._reload_interval = reload_interval

        self._models: dict[str, ModelConfig] = {}
        self._complexity_high: set[str] = set()
        self._complexity_low: set[str] = set()
        self._lock = threading.RLock()
        self._last_modified: float = 0
        self._callbacks: list[Callable[[dict[str, ModelConfig]], None]] = []
        self._shutdown_event = threading.Event()
        self._watcher_thread: threading.Thread | None = None

        # Load initial config
        self._load_config()

        # Start auto-reload watcher if enabled
        if auto_reload and config_file.exists():
            self._start_watcher()

    def shutdown(self):
        """Gracefully shutdown the config watcher thread."""
        self._shutdown_event.set()
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=5.0)
            if self._watcher_thread.is_alive():
                logger.warning("Config watcher thread did not stop gracefully")

    def _load_config(self) -> bool:
        """Load configuration from file. Returns True if loaded successfully."""
        try:
            if not self._config_file.exists():
                logger.warning(
                    f"Config file not found: {self._config_file}, using defaults"
                )
                self._load_defaults()
                return False

            # Check if file has been modified
            mtime = self._config_file.stat().st_mtime
            if mtime <= self._last_modified:
                return False  # No changes

            with open(self._config_file) as f:
                data = json.load(f)

            with self._lock:
                # Parse models
                self._models = {}
                for name, config in data.get("models", {}).items():
                    try:
                        self._models[name] = ModelConfig(
                            model_id=config["model_id"],
                            tier=config["tier"],
                            max_tokens=config.get("max_tokens", 2048),
                            context_window=config.get("context_window", 4096),
                            avg_latency_ms=config.get("avg_latency_ms", 100),
                            task_types=config.get("task_types", ["chat"]),
                        )
                    except Exception as e:
                        logger.error(f"Failed to parse model config '{name}': {e}")

                # Parse complexity keywords
                keywords = data.get("complexity_keywords", {})
                self._complexity_high = set(keywords.get("high", []))
                self._complexity_low = set(keywords.get("low", []))

                self._last_modified = mtime

            logger.info(
                f"Loaded {len(self._models)} model configs from {self._config_file}"
            )

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(self._models.copy())
                except Exception as e:
                    logger.error(f"Config reload callback error: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            self._load_defaults()
            return False

    def _load_defaults(self):
        """Load hardcoded defaults as fallback."""
        with self._lock:
            self._models = {
                "qwen2.5-3b": ModelConfig(
                    model_id="Qwen/Qwen2.5-3B-Instruct",
                    tier=ModelTier.STANDARD,
                    max_tokens=4096,
                    context_window=8192,
                    avg_latency_ms=200,
                    task_types=[TaskType.CHAT, TaskType.REASONING, TaskType.CODE],
                ),
            }
            self._complexity_high = {"analyze", "compare", "implement", "complex"}
            self._complexity_low = {"what is", "simple", "translate", "hello"}

    def _start_watcher(self):
        """Start background thread to watch for config changes."""

        def watcher():
            while not self._shutdown_event.is_set():
                try:
                    # Use wait with timeout instead of sleep for graceful shutdown
                    if self._shutdown_event.wait(timeout=self._reload_interval):
                        break  # Shutdown requested
                    self._load_config()
                except Exception as e:
                    logger.error(f"Config watcher error: {e}")

        self._watcher_thread = threading.Thread(
            target=watcher, daemon=True, name="ConfigWatcher"
        )
        self._watcher_thread.start()
        logger.info(f"Started config watcher (interval: {self._reload_interval}s)")

    def reload(self) -> bool:
        """Manually trigger config reload. Returns True if config was updated."""
        self._last_modified = 0  # Force reload
        return self._load_config()

    def get_models(self) -> dict[str, ModelConfig]:
        """Get all model configurations (thread-safe copy)."""
        with self._lock:
            return self._models.copy()

    def get_model(self, name: str) -> ModelConfig | None:
        """Get a specific model configuration."""
        with self._lock:
            return self._models.get(name)

    def get_complexity_keywords(self) -> tuple:
        """Get complexity keyword sets."""
        with self._lock:
            return self._complexity_high.copy(), self._complexity_low.copy()

    def on_reload(self, callback: Callable[[dict[str, ModelConfig]], None]):
        """Register a callback for when config is reloaded."""
        self._callbacks.append(callback)


# Global singleton
_config_loader: ModelConfigLoader | None = None
_config_loader_lock = threading.Lock()


def get_model_config_loader() -> ModelConfigLoader:
    """Get the global model config loader (thread-safe singleton)."""
    global _config_loader
    if _config_loader is None:
        with _config_loader_lock:
            if _config_loader is None:
                _config_loader = ModelConfigLoader()
    return _config_loader


def reload_model_config() -> bool:
    """Reload model configuration. Returns True if updated."""
    return get_model_config_loader().reload()


__all__ = [
    "ModelConfig",
    "ModelConfigLoader",
    "ModelTier",
    "TaskType",
    "get_model_config_loader",
    "reload_model_config",
]
