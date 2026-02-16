"""
Early Stop Heuristics (Principle L)
===================================
Stop generation when 3 consecutive sentences end in period.

Strategy:
- Monitor generation output token-by-token
- Detect sentence boundaries using punctuation patterns
- Stop early when content is complete
- Save compute cycles and reduce latency

Reference: "Stop generation when 3 consecutive sentences end in period"
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class StopReason(Enum):
    """Reasons for stopping generation."""

    MAX_TOKENS = "max_tokens"
    EOS_TOKEN = "eos_token"
    EARLY_STOP = "early_stop"
    STOP_SEQUENCE = "stop_sequence"
    TIMEOUT = "timeout"

@dataclass
class EarlyStopConfig:
    """Configuration for early stop heuristics."""

    # Sentence ending detection
    consecutive_sentences_threshold: int = 3
    sentence_end_pattern: str = r"[.!?]"

    # Minimum content requirements
    min_sentences: int = 1
    min_tokens: int = 20
    min_characters: int = 100

    # Stop sequences
    stop_sequences: list[str] = field(
        default_factory=lambda: [
            "\n\n\n",  # Triple newline
            "---",  # Section break
            "###",  # End marker
            "[END]",
            "</response>",
        ]
    )

    # Quality checks
    require_complete_sentence: bool = True

    # Task-specific overrides
    task_configs: dict = field(
        default_factory=lambda: {
            "simplify": {"consecutive_sentences_threshold": 3, "min_sentences": 1},
            "explain": {"consecutive_sentences_threshold": 4, "min_sentences": 2},
            "summarize": {"consecutive_sentences_threshold": 2, "min_sentences": 1},
            "translate": {"consecutive_sentences_threshold": 1, "min_sentences": 1},
            "qa": {"consecutive_sentences_threshold": 2, "min_sentences": 1},
        }
    )

@dataclass
class GenerationState:
    """Tracks state during generation for early stop detection."""

    generated_text: str = ""
    token_count: int = 0
    sentence_count: int = 0
    consecutive_complete_sentences: int = 0
    last_sentence_end_position: int = 0
    should_stop: bool = False
    stop_reason: StopReason | None = None

    def reset(self):
        """Reset state for new generation."""
        self.generated_text = ""
        self.token_count = 0
        self.sentence_count = 0
        self.consecutive_complete_sentences = 0
        self.last_sentence_end_position = 0
        self.should_stop = False
        self.stop_reason = None

class EarlyStopDetector:
    """
    Detects when generation can be stopped early.

    Uses heuristics to determine when generated content is complete:
    - Monitors consecutive sentence endings
    - Checks for stop sequences
    - Validates minimum content requirements
    """

    def __init__(self, config: EarlyStopConfig | None = None, task: str = "simplify"):
        self.base_config = config or EarlyStopConfig()
        self.task = task
        self.config = self._get_task_config(task)
        self.state = GenerationState()

        # Compile regex patterns
        self._sentence_end_re = re.compile(self.config.sentence_end_pattern)
        self._incomplete_patterns = re.compile(
            r"[,;:]$|(?:and|but|or|so|yet|because|although|while|if|when)$",
            re.IGNORECASE,
        )

    def _get_task_config(self, task: str) -> EarlyStopConfig:
        """Get task-specific configuration."""
        task_overrides = self.base_config.task_configs.get(task, {})

        # Create new config with task-specific overrides
        config_dict = {
            "consecutive_sentences_threshold": self.base_config.consecutive_sentences_threshold,
            "sentence_end_pattern": self.base_config.sentence_end_pattern,
            "min_sentences": self.base_config.min_sentences,
            "min_tokens": self.base_config.min_tokens,
            "min_characters": self.base_config.min_characters,
            "stop_sequences": self.base_config.stop_sequences,
            "require_complete_sentence": self.base_config.require_complete_sentence,
            "task_configs": self.base_config.task_configs,
        }
        config_dict.update(task_overrides)

        return EarlyStopConfig(**config_dict)

    def reset(self):
        """Reset detector state for new generation."""
        self.state.reset()

    def check_token(self, new_token: str) -> bool:
        """
        Check if generation should stop after new token.

        Args:
            new_token: Newly generated token

        Returns:
            True if generation should stop
        """
        self.state.generated_text += new_token
        self.state.token_count += 1

        # Check stop sequences first
        if self._check_stop_sequences():
            self.state.should_stop = True
            self.state.stop_reason = StopReason.STOP_SEQUENCE
            return True

        # Check for sentence endings
        self._update_sentence_tracking()

        # Check early stop conditions
        if self._should_early_stop():
            self.state.should_stop = True
            self.state.stop_reason = StopReason.EARLY_STOP
            return True

        return False

    def check_text(self, text: str) -> bool:
        """
        Check if complete text warrants early stop.

        For non-streaming scenarios where we check periodically.

        Args:
            text: Complete generated text so far

        Returns:
            True if generation should stop
        """
        self.state.generated_text = text
        self.state.token_count = len(text.split())  # Approximate

        # Check stop sequences
        if self._check_stop_sequences():
            self.state.should_stop = True
            self.state.stop_reason = StopReason.STOP_SEQUENCE
            return True

        # Update sentence tracking
        self._update_sentence_tracking()

        # Check early stop
        if self._should_early_stop():
            self.state.should_stop = True
            self.state.stop_reason = StopReason.EARLY_STOP
            return True

        return False

    def _check_stop_sequences(self) -> bool:
        """Check if any stop sequence is present."""
        text = self.state.generated_text
        for seq in self.config.stop_sequences:
            if seq in text:
                logger.debug("Stop sequence detected: %s", seq)
                return True
        return False

    def _update_sentence_tracking(self):
        """Update sentence counting and consecutive tracking."""
        text = self.state.generated_text

        # Find all sentence endings
        endings = list(self._sentence_end_re.finditer(text))
        self.state.sentence_count = len(endings)

        if not endings:
            self.state.consecutive_complete_sentences = 0
            return

        # Check consecutive complete sentences at the end
        consecutive = 0
        for match in reversed(endings):
            end_pos = match.end()

            # Check if there's significant text after this ending
            text_after = text[end_pos:].strip()

            if len(text_after) < 3:  # At or near end of text
                consecutive += 1
            else:
                # There's more text after this sentence
                consecutive = 0
                break

        self.state.consecutive_complete_sentences = consecutive

        if endings:
            self.state.last_sentence_end_position = endings[-1].end()

    def _should_early_stop(self) -> bool:
        """Check if early stop conditions are met."""
        # Check minimum requirements
        if self.state.sentence_count < self.config.min_sentences:
            return False

        if self.state.token_count < self.config.min_tokens:
            return False

        if len(self.state.generated_text) < self.config.min_characters:
            return False

        # Check consecutive sentences threshold
        if (
            self.state.consecutive_complete_sentences
            >= self.config.consecutive_sentences_threshold
        ):
            # Verify last sentence is complete
            if self.config.require_complete_sentence:
                text_end = self.state.generated_text.rstrip()
                if self._incomplete_patterns.search(text_end):
                    return False

            logger.debug(
                f"Early stop triggered: {self.state.consecutive_complete_sentences} "
                f"consecutive complete sentences"
            )
            return True

        return False

    def get_clean_output(self) -> str:
        """
        Get cleaned output text after early stop.

        Removes any partial content after the last complete sentence.
        """
        text = self.state.generated_text

        # Remove stop sequences
        for seq in self.config.stop_sequences:
            if seq in text:
                text = text.split(seq)[0]

        # Find last complete sentence
        endings = list(self._sentence_end_re.finditer(text))
        if endings:
            last_end = endings[-1].end()
            # Keep any closing punctuation or quotes
            remaining = text[last_end:].strip()
            if remaining and remaining[0] in "\"')":
                last_end = text.find(remaining[0], last_end) + 1
            text = text[:last_end].strip()

        return text

    def get_state(self) -> dict:
        """Get current state as dictionary."""
        return {
            "token_count": self.state.token_count,
            "sentence_count": self.state.sentence_count,
            "consecutive_complete_sentences": self.state.consecutive_complete_sentences,
            "should_stop": self.state.should_stop,
            "stop_reason": self.state.stop_reason.value
            if self.state.stop_reason
            else None,
            "text_length": len(self.state.generated_text),
        }

class StreamingEarlyStopCallback:
    """
    Callback for streaming generation with early stop support.

    Integrates with vLLM and transformers streaming generation.
    """

    def __init__(self, detector: EarlyStopDetector, on_stop: Callable | None = None):
        self.detector = detector
        self.on_stop = on_stop
        self._accumulated = ""

    def __call__(self, token: str) -> bool:
        """
        Process new token and return whether to continue.

        Args:
            token: New token from generation

        Returns:
            True to continue, False to stop
        """
        should_stop = self.detector.check_token(token)

        if should_stop and self.on_stop:
            self.on_stop(self.detector.get_clean_output())

        return not should_stop

    def get_result(self) -> str:
        """Get final cleaned result."""
        return self.detector.get_clean_output()

def create_early_stop_detector(task: str = "simplify") -> EarlyStopDetector:
    """
    Create early stop detector for given task.

    Args:
        task: Task type (simplify, explain, summarize, translate, qa)

    Returns:
        Configured EarlyStopDetector
    """
    return EarlyStopDetector(task=task)

def check_should_stop(text: str, task: str = "simplify") -> bool:
    """
    Quick check if text generation should stop.

    Args:
        text: Generated text
        task: Task type

    Returns:
        True if generation should stop
    """
    detector = create_early_stop_detector(task)
    return detector.check_text(text)

def get_clean_output(text: str, task: str = "simplify") -> str:
    """
    Clean generated text by removing incomplete content.

    Args:
        text: Raw generated text
        task: Task type

    Returns:
        Cleaned text
    """
    detector = create_early_stop_detector(task)
    detector.check_text(text)
    return detector.get_clean_output()
