"""
Policy Module - Unified Content Policy Engine
==============================================

Consolidates all hardcoded content filtering, content alignment,
safety guards, and alignment logic into a single configurable module.

This module extracts and centralizes the following hardcoded behaviors:
1. Safety filtering (harmful content detection, jailbreak prevention)
2. Content moderation (content_domain alignment, complexity-level adaptation)
3. User/teacher role-based restrictions
4. Secret/PII redaction
5. Response modification and rejection logic

All behaviors are now configurable via environment variables and the
policy configuration file (policy/config.default.json).

CRITICAL: When ALLOW_UNRESTRICTED_MODE=true:
- All content_domain/student-specific filters are bypassed
- Model outputs pass through unchanged
- System-level safety (secrets, PII) can still be enabled separately
- Local-only execution is enforced unless ALLOW_EXTERNAL_CALLS=true
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PolicyMode(str, Enum):
    """Operating mode for the policy engine."""

    RESTRICTED = "restricted"  # Full policy enforcement
    UNRESTRICTED = "unrestricted"  # Bypass content/moderation filters
    EXTERNAL_ALLOWED = "external_allowed"  # Unrestricted + external calls allowed
    OPEN = "open"  # Default - open with essential safety
    MODERATED = "moderated"  # Moderated with content_domain
    RESEARCH = "research"  # Maximum freedom for academic work


class RiskLevel(str, Enum):
    """Risk levels for detected content."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PolicyCheckResult:
    """Result of a policy check."""

    allowed: bool
    blocked: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    issues: list[str] = field(default_factory=list)
    filtered_content: str | None = None
    rejection_message: str | None = None
    policy_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "blocked": self.blocked,
            "risk_level": self.risk_level.value,
            "issues": self.issues,
            "policy_applied": self.policy_applied,
        }


@dataclass
class PolicyConfig:
    """Configuration for policy engine loaded from config file and env vars."""

    # Master toggles (from environment)
    allow_unrestricted_mode: bool = False
    policy_filters_enabled: bool = True
    sensitive_response_blocking: bool = True
    content_moderation: bool = True
    complexity_adaptation: bool = True

    # External call controls
    allow_external_calls: bool = False
    external_call_audit: bool = True
    external_calls_log_path: str = "logs/external_calls.log"

    # Safety settings
    block_harmful_content: bool = True
    detect_jailbreaks: bool = True
    redact_secrets: bool = True
    redact_pii: bool = True
    max_input_length: int = 50000

    # Curriculum settings (only apply when content_moderation=True)
    content_quality_threshold: float = 0.70
    require_complexity_match: bool = False
    allowed_grade_range: tuple[int, int] = (1, 12)

    # Content patterns
    harmful_patterns: list[dict[str, Any]] = field(default_factory=list)
    jailbreak_patterns: list[dict[str, Any]] = field(default_factory=list)
    content_context_patterns: list[str] = field(default_factory=list)

    # Response modification
    add_ucontent_domainainty_disclaimers: bool = True
    add_source_verification_prompts: bool = True

    @classmethod
    def from_env_and_file(cls, config_path: Path | None = None) -> "PolicyConfig":
        """Load configuration from environment variables and config file."""
        config = cls()

        # Check Settings.UNIVERSAL_MODE first (new unified approach)
        try:
            from ..core.config import get_settings

            universal_mode = get_settings().UNIVERSAL_MODE
        except Exception:
            universal_mode = True  # Default to universal mode

        # Load from environment variables (highest priority)
        # DEFAULT: OPEN mode - unrestricted but safe
        # When UNIVERSAL_MODE=true, always enable unrestricted mode
        config.allow_unrestricted_mode = (
            universal_mode
            or os.getenv(
                "ALLOW_UNRESTRICTED_MODE",
                "true",  # Default to open
            ).lower()
            == "true"
        )

        config.policy_filters_enabled = (
            os.getenv(
                "POLICY_FILTERS_ENABLED",
                "false",  # Default to disabled
            ).lower()
            == "true"
        )

        config.sensitive_response_blocking = (
            os.getenv(
                "SENSITIVE_RESPONSE_BLOCKING",
                "false",  # Default to disabled
            ).lower()
            == "true"
        )

        # UNIVERSAL_MODE disables content_domain enforcement
        config.content_moderation = (
            not universal_mode
            and os.getenv(
                "CONTENT_MODERATION",
                "false",  # Default to disabled
            ).lower()
            == "true"
        )

        config.complexity_adaptation = (
            not universal_mode
            and os.getenv(
                "COMPLEXITY_ADAPTATION",
                "false",  # Optional, default off
            ).lower()
            == "true"
        )

        config.allow_external_calls = (
            os.getenv(
                "ALLOW_EXTERNAL_CALLS",
                "true",  # Default to allowed
            ).lower()
            == "true"
        )

        config.external_call_audit = (
            os.getenv("EXTERNAL_CALL_AUDIT", "true").lower() == "true"
        )

        # SAFETY: Always block genuinely harmful content
        config.block_harmful_content = (
            os.getenv(
                "BLOCK_HARMFUL_CONTENT",
                "true",  # Always on for safety
            ).lower()
            == "true"
        )

        config.detect_jailbreaks = (
            os.getenv(
                "DETECT_JAILBREAKS",
                "false",  # Disabled - trust users
            ).lower()
            == "true"
        )

        config.redact_secrets = (
            os.getenv(
                "REDACT_SECRETS",
                "true",  # Keep for security
            ).lower()
            == "true"
        )

        config.redact_pii = (
            os.getenv(
                "REDACT_PII",
                "false",  # Disabled by default
            ).lower()
            == "true"
        )

        # Load from config file if available
        if config_path is None:
            # Try default locations
            for path in [
                Path("policy/config.default.json"),
                Path(__file__).parent.parent.parent / "policy" / "config.default.json",
            ]:
                if path.exists():
                    config_path = path
                    break

        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = json.load(f)

                # Load patterns from file (don't override env vars)
                config.harmful_patterns = file_config.get("harmful_patterns", [])
                config.jailbreak_patterns = file_config.get("jailbreak_patterns", [])
                config.content_context_patterns = file_config.get(
                    "content_context_patterns", []
                )

                # Load content_domain settings from file if not set via env
                content_domain = file_config.get("content_domain", {})
                if "content_quality_threshold" in content_domain:
                    config.content_quality_threshold = content_domain[
                        "content_quality_threshold"
                    ]

                logger.info(f"Loaded policy config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load policy config from {config_path}: {e}")

        return config


class PolicyEngine:
    """
    Unified policy engine that consolidates all content filtering logic.

    This engine replaces hardcoded checks scattered throughout the codebase
    with a single, configurable policy enforcement point.

    When ALLOW_UNRESTRICTED_MODE is enabled:
    - Content moderation is bypassed
    - Grade-level adaptation is skipped
    - Content context filtering is disabled
    - User/teacher role restrictions are removed

    System-level safety (secrets, PII) can still be enabled independently.
    """

    def __init__(self, config: PolicyConfig | None = None):
        """Initialize the policy engine."""
        self.config = config or PolicyConfig.from_env_and_file()
        self._compiled_harmful_patterns: list[tuple[re.Pattern, RiskLevel]] = []
        self._compiled_jailbreak_patterns: list[tuple[re.Pattern, RiskLevel]] = []
        self._compiled_content_patterns: list[re.Pattern] = []
        self._lock = threading.Lock()
        self._initialized = False
        self._external_call_log = None

        # Statistics
        self._stats = {
            "checks_performed": 0,
            "content_blocked": 0,
            "content_filtered": 0,
            "policies_bypassed": 0,
            "mode_switches": 0,
        }

    @property
    def mode(self) -> PolicyMode:
        """Get current operating mode."""
        # Map config settings to simplified modes
        if self.config.content_moderation:
            return PolicyMode.MODERATED
        elif self.config.allow_unrestricted_mode:
            if self.config.allow_external_calls:
                return PolicyMode.RESEARCH
            return PolicyMode.OPEN
        return PolicyMode.RESTRICTED

    @property
    def mode_description(self) -> str:
        """Get human-readable mode description."""
        mode = self.mode
        descriptions = {
            PolicyMode.OPEN: "Open AI for AI, research & noble purposes",
            PolicyMode.MODERATED: "Education mode with content domain alignment",
            PolicyMode.RESEARCH: "Research mode with maximum freedom",
            PolicyMode.RESTRICTED: "Restricted mode with full policy enforcement",
            PolicyMode.UNRESTRICTED: "Unrestricted mode - filters bypassed",
            PolicyMode.EXTERNAL_ALLOWED: "Unrestricted with external calls allowed",
        }
        return descriptions.get(mode, "Unknown mode")

    def switch_mode(self, new_mode: PolicyMode) -> dict[str, Any]:
        """
        Switch to a different policy mode at runtime.

        Args:
            new_mode: The PolicyMode to switch to

        Returns:
            Dict with old mode, new mode, and settings applied
        """
        old_mode = self.mode

        with self._lock:
            # Define mode configurations to avoid code duplication
            unrestricted_config = {
                "allow_unrestricted_mode": True,
                "policy_filters_enabled": False,
                "content_moderation": False,
                "allow_external_calls": True,
                "detect_jailbreaks": False,
            }
            restricted_config = {
                "allow_unrestricted_mode": False,
                "policy_filters_enabled": True,
                "content_moderation": True,
                "allow_external_calls": False,
                "detect_jailbreaks": True,
            }

            # Map modes to their configurations
            mode_configs = {
                PolicyMode.OPEN: unrestricted_config,
                PolicyMode.RESEARCH: unrestricted_config,
                PolicyMode.MODERATED: restricted_config,
                PolicyMode.RESTRICTED: restricted_config,
            }

            config_to_apply = mode_configs.get(new_mode, unrestricted_config)
            for key, value in config_to_apply.items():
                setattr(self.config, key, value)

            # Safety settings always remain on
            self.config.block_harmful_content = True
            self.config.redact_secrets = True

            self._stats["mode_switches"] += 1

        logger.info(f"Policy mode switched: {old_mode.value} -> {new_mode.value}")

        return {
            "old_mode": old_mode.value,
            "new_mode": self.mode.value,
            "description": self.mode_description,
            "settings": {
                "unrestricted_mode": self.config.allow_unrestricted_mode,
                "policy_filters": self.config.policy_filters_enabled,
                "content_moderation": self.config.content_moderation,
                "external_calls": self.config.allow_external_calls,
                "harmful_content_blocking": self.config.block_harmful_content,
            },
        }

    def _ensure_initialized(self):
        """Lazy initialization of compiled patterns."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Compile harmful patterns
            default_harmful = [
                # Violence - extracted from safety.py HARMFUL_PATTERNS
                {
                    "pattern": r"(?i)\b(how to|ways to|instructions for)\s+(kill|harm|hurt|attack|murder)\b",
                    "risk": "critical",
                    "origin": "safety.py:218",
                },
                {
                    "pattern": r"(?i)\b(make|build|create)\s+(bomb|explosive|weapon)\b",
                    "risk": "critical",
                    "origin": "safety.py:219",
                },
                # Illegal activities
                {
                    "pattern": r"(?i)\b(hack|crack|steal)\s+(password|account|credit card)\b",
                    "risk": "high",
                    "origin": "safety.py:222",
                },
                {
                    "pattern": r"(?i)\b(create|generate)\s+(malware|virus|ransomware)\b",
                    "risk": "critical",
                    "origin": "safety.py:223",
                },
            ]

            for pattern_def in self.config.harmful_patterns or default_harmful:
                try:
                    pattern = re.compile(pattern_def["pattern"])
                    risk = RiskLevel(pattern_def.get("risk", "high"))
                    self._compiled_harmful_patterns.append((pattern, risk))
                except Exception as e:
                    logger.warning(f"Failed to compile harmful pattern: {e}")

            # Compile jailbreak patterns
            default_jailbreak = [
                # Extracted from safety.py HARMFUL_PATTERNS
                {
                    "pattern": r"(?i)(ignore|forget|disregard)\s+(previous|all)\s+(instructions|rules|guidelines)",
                    "risk": "high",
                    "origin": "safety.py:226",
                },
                {
                    "pattern": r"(?i)pretend\s+you\s+(are|can|have)\s+no\s+(restrictions|limits|rules)",
                    "risk": "high",
                    "origin": "safety.py:227",
                },
                {
                    "pattern": r"(?i)DAN\s+mode|do\s+anything\s+now",
                    "risk": "high",
                    "origin": "safety.py:228",
                },
            ]

            for pattern_def in self.config.jailbreak_patterns or default_jailbreak:
                try:
                    pattern = re.compile(pattern_def["pattern"])
                    risk = RiskLevel(pattern_def.get("risk", "high"))
                    self._compiled_jailbreak_patterns.append((pattern, risk))
                except Exception as e:
                    logger.warning(f"Failed to compile jailbreak pattern: {e}")

            # Compile educational context patterns
            default_educational = [
                # Extracted from safety.py CONTENT_PATTERNS
                r"(?i)(explain|understand|learn about|history of|science of)",
                r"(?i)(for legitimate purposes|in theory|academically)",
                r"(?i)(what is|how does|why does)",
            ]

            for pattern_str in (
                self.config.content_context_patterns or default_educational
            ):
                try:
                    self._compiled_content_patterns.append(re.compile(pattern_str))
                except Exception as e:
                    logger.warning(f"Failed to compile educational pattern: {e}")

            # Initialize external call log if needed
            if self.config.external_call_audit:
                log_path = Path(self.config.external_calls_log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)

            self._initialized = True
            logger.info(f"PolicyEngine initialized in {self.mode.value} mode")

    def apply_input_policy(
        self, text: str, context: dict[str, Any] | None = None
    ) -> PolicyCheckResult:
        """
        Apply policy checks to input text.

        Args:
            text: Input text to check
            context: Optional context (user_id, complexity_level, etc.)

        Returns:
            PolicyCheckResult with check outcome
        """
        self._ensure_initialized()
        self._stats["checks_performed"] += 1

        context = context or {}
        issues = []
        policy_applied = []
        max_risk = RiskLevel.LOW
        filtered_text = text

        # In unrestricted mode, only apply system-level safety
        if self.config.allow_unrestricted_mode:
            self._stats["policies_bypassed"] += 1
            policy_applied.append("UNRESTRICTED_MODE_ACTIVE")

            # Still apply secrets/PII redaction if enabled
            if self.config.redact_secrets or self.config.redact_pii:
                filtered_text, redaction_info = self._apply_redaction(text)
                if redaction_info:
                    policy_applied.append("secrets_redaction")

            return PolicyCheckResult(
                allowed=True,
                blocked=False,
                risk_level=RiskLevel.LOW,
                filtered_content=filtered_text,
                policy_applied=policy_applied,
            )

        # === RESTRICTED MODE: Full policy enforcement ===

        # 1. Input length check
        if len(text) > self.config.max_input_length:
            filtered_text = (
                text[: self.config.max_input_length] + "\n[Content truncated]"
            )
            issues.append("Input truncated due to length")
            policy_applied.append("length_limit")

        # 2. Harmful content detection
        if self.config.block_harmful_content:
            for pattern, risk in self._compiled_harmful_patterns:
                if pattern.search(text):
                    # Check for AIal context exemption
                    is_legitimate = any(
                        ep.search(text) for ep in self._compiled_content_patterns
                    )
                    if not is_legitimate:
                        issues.append("Potentially harmful content detected")
                        if risk.value > max_risk.value:
                            max_risk = risk
                        policy_applied.append("harmful_content_filter")

        # 3. Jailbreak detection
        if self.config.detect_jailbreaks:
            for pattern, risk in self._compiled_jailbreak_patterns:
                if pattern.search(text):
                    issues.append("Jailbreak attempt detected")
                    if risk.value > max_risk.value:
                        max_risk = risk
                    policy_applied.append("jailbreak_detection")

        # 4. Secrets/PII redaction
        if self.config.redact_secrets or self.config.redact_pii:
            filtered_text, redaction_info = self._apply_redaction(filtered_text)
            if redaction_info:
                policy_applied.append("secrets_pii_redaction")

        # Determine if content should be blocked
        blocked = max_risk in [RiskLevel.CRITICAL, RiskLevel.HIGH]

        if blocked:
            self._stats["content_blocked"] += 1
            rejection_message = self._get_rejection_message(max_risk)
        else:
            rejection_message = None

        if issues:
            self._stats["content_filtered"] += 1

        return PolicyCheckResult(
            allowed=not blocked,
            blocked=blocked,
            risk_level=max_risk,
            issues=issues,
            filtered_content=filtered_text,
            rejection_message=rejection_message,
            policy_applied=policy_applied,
        )

    def apply_output_policy(
        self, text: str, context: dict[str, Any] | None = None
    ) -> str:
        """
        Apply policy transformations to model output.

        In unrestricted mode, returns text unchanged.
        In restricted mode, applies filtering and modifications.

        Args:
            text: Model output text
            context: Optional context

        Returns:
            Filtered/modified text
        """
        self._ensure_initialized()

        # In unrestricted mode, pass through unchanged
        if self.config.allow_unrestricted_mode:
            return text

        # Apply output filtering
        filtered = text

        # Remove potentially dangerous code patterns (from safety.py filter_response)
        if self.config.sensitive_response_blocking:
            filtered = re.sub(
                r"(?i)(os\.system|subprocess|eval|exec)\s*\([^)]+\)",
                "[CODE_REMOVED_BY_POLICY]",
                filtered,
            )

        # Redact secrets in output
        if self.config.redact_secrets:
            filtered, _ = self._apply_redaction(filtered)

        return filtered

    def check_content_domain_alignment(
        self,
        text: str,
        complexity_level: int | None = None,
        subject: str | None = None,
    ) -> PolicyCheckResult:
        """
        Check content alignment with content_domain standards.

        In unrestricted mode, always returns allowed=True.

        Args:
            text: Content to check (used for alignment analysis)
            complexity_level: Target complexity level (1-12)
            subject: Subject area (reserved for subject-specific checks)

        Returns:
            PolicyCheckResult
        """
        self._ensure_initialized()

        # These parameters are validated but only used when content_domain enforcement is enabled
        _ = (text, subject)  # Suppress unused parameter warnings for bypass case

        # In unrestricted mode, skip content checks
        if (
            self.config.allow_unrestricted_mode
            or not self.config.content_moderation
        ):
            return PolicyCheckResult(
                allowed=True,
                blocked=False,
                policy_applied=["content_domain_check_bypassed"],
            )

        # Grade level validation
        if complexity_level is not None:
            min_grade, max_grade = self.config.allowed_grade_range
            if not (min_grade <= complexity_level <= max_grade):
                return PolicyCheckResult(
                    allowed=False,
                    blocked=True,
                    issues=[
                        f"Grade level {complexity_level} outside allowed range {min_grade}-{max_grade}"
                    ],
                    policy_applied=["complexity_level_validation"],
                )

        return PolicyCheckResult(
            allowed=True,
            blocked=False,
            policy_applied=["content_domain_check_passed"],
        )

    def log_external_call(
        self, endpoint: str, method: str = "GET", payload: dict | None = None
    ):
        """Log an external API call for audit purposes."""
        if not self.config.external_call_audit:
            return

        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "endpoint": endpoint,
            "method": method,
            "payload_size": len(json.dumps(payload)) if payload else 0,
        }

        log_path = Path(self.config.external_calls_log_path)
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log external call: {e}")

    def can_make_external_call(self, endpoint: str) -> bool:
        """Check if external calls are allowed and log if enabled."""
        if not self.config.allow_external_calls:
            logger.warning(
                f"External call to {endpoint} blocked (ALLOW_EXTERNAL_CALLS=false)"
            )
            return False

        # Log the call attempt
        self.log_external_call(endpoint, "CHECK")
        return True

    def _apply_redaction(self, text: str) -> tuple[str, list[dict]]:
        """Apply secrets and PII redaction to text."""
        redaction_info = []
        result = text

        # Secret patterns (extracted from safety.py SecretScanner)
        secret_patterns = [
            (r"AKIA[0-9A-Z]{16}", "AWS_KEY"),
            (r"sk-[A-Za-z0-9]{48}", "OPENAI_KEY"),
            (r"ghp_[A-Za-z0-9]{36}", "GITHUB_TOKEN"),
            (
                r'(?i)(password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^\s"\']{8,})',
                "PASSWORD",
            ),
            (r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*", "JWT"),
        ]

        for pattern_str, secret_type in secret_patterns:
            pattern = re.compile(pattern_str)
            matches = list(pattern.finditer(result))
            for match in reversed(matches):
                redaction_info.append(
                    {
                        "type": secret_type,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
                result = (
                    result[: match.start()]
                    + f"[REDACTED_{secret_type}]"
                    + result[match.end() :]
                )

        # PII patterns
        if self.config.redact_pii:
            pii_patterns = [
                (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
                (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "EMAIL"),
            ]
            for pattern_str, pii_type in pii_patterns:
                pattern = re.compile(pattern_str)
                matches = list(pattern.finditer(result))
                for match in reversed(matches):
                    redaction_info.append(
                        {
                            "type": pii_type,
                            "start": match.start(),
                        }
                    )
                    result = (
                        result[: match.start()]
                        + f"[REDACTED_{pii_type}]"
                        + result[match.end() :]
                    )

        return result, redaction_info

    def _get_rejection_message(self, risk_level: RiskLevel) -> str:
        """Get appropriate rejection message based on risk level."""
        # Extracted from safety.py get_rejection_message
        if risk_level == RiskLevel.CRITICAL:
            return (
                "I'm sorry, but I can't help with that request as it may involve "
                "harmful or dangerous content. If you're researching a topic for "
                "legitimate purposes, please rephrase your question to clarify the context."
            )
        elif risk_level == RiskLevel.HIGH:
            return (
                "I'm unable to assist with this particular request. If you have a "
                "legitimate educational or professional need, please provide more "
                "context about what you're trying to accomplish."
            )
        else:
            return (
                "I noticed some concerns with your request. Could you please "
                "rephrase it so I can better assist you?"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get policy engine statistics."""
        return {
            **self._stats,
            "mode": self.mode.value,
            "config": {
                "unrestricted_mode": self.config.allow_unrestricted_mode,
                "filters_enabled": self.config.policy_filters_enabled,
                "content_moderation": self.config.content_moderation,
                "external_calls_allowed": self.config.allow_external_calls,
            },
        }


# Singleton instance
_policy_engine: PolicyEngine | None = None
_policy_lock = threading.Lock()


def get_policy_engine() -> PolicyEngine:
    """Get or create the global policy engine instance."""
    global _policy_engine

    if _policy_engine is None:
        with _policy_lock:
            if _policy_engine is None:
                _policy_engine = PolicyEngine()

    return _policy_engine


def print_startup_banner():
    """Print startup banner showing current policy mode."""
    engine = get_policy_engine()
    mode = engine.mode
    config = engine.config

    # Colors for terminal
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    _CYAN = "\033[96m"  # Available for future use
    RESET = "\033[0m"
    BOLD = "\033[1m"

    banner_lines = [
        "",
        f"{BOLD}╔══════════════════════════════════════════════════════════════════╗{RESET}",
        f"{BOLD}║                    SHIKSHA SETU POLICY ENGINE                     ║{RESET}",
        f"{BOLD}╠══════════════════════════════════════════════════════════════════╣{RESET}",
    ]

    if mode == PolicyMode.UNRESTRICTED:
        mode_color = YELLOW
        mode_str = "UNRESTRICTED"
        mode_desc = "All content/moderation filters BYPASSED"
    elif mode == PolicyMode.EXTERNAL_ALLOWED:
        mode_color = RED
        mode_str = "UNRESTRICTED + EXTERNAL CALLS"
        mode_desc = "Filters bypassed, external API calls ALLOWED"
    else:
        mode_color = GREEN
        mode_str = "RESTRICTED (DEFAULT)"
        mode_desc = "Full policy enforcement active"

    banner_lines.extend(
        [
            f"{BOLD}║{RESET}  Mode: {mode_color}{BOLD}{mode_str:^52}{RESET}  {BOLD}║{RESET}",
            f"{BOLD}║{RESET}  {mode_desc:^62}  {BOLD}║{RESET}",
            f"{BOLD}╠══════════════════════════════════════════════════════════════════╣{RESET}",
            f"{BOLD}║{RESET}  Policy Filters:      {'ENABLED' if config.policy_filters_enabled else 'DISABLED':>10}                            {BOLD}║{RESET}",
            f"{BOLD}║{RESET}  Curriculum Enforce:  {'ENABLED' if config.content_moderation else 'DISABLED':>10}                            {BOLD}║{RESET}",
            f"{BOLD}║{RESET}  Harmful Content:     {'BLOCKED' if config.block_harmful_content else 'ALLOWED':>10}                            {BOLD}║{RESET}",
            f"{BOLD}║{RESET}  Secret Redaction:    {'ENABLED' if config.redact_secrets else 'DISABLED':>10}                            {BOLD}║{RESET}",
            f"{BOLD}║{RESET}  External Calls:      {'ALLOWED' if config.allow_external_calls else 'BLOCKED':>10}                            {BOLD}║{RESET}",
            f"{BOLD}╚══════════════════════════════════════════════════════════════════╝{RESET}",
            "",
        ]
    )

    for line in banner_lines:
        print(line)

    # Also log it
    logger.info(f"Policy Engine Mode: {mode.value}")
    logger.info(f"  - Unrestricted: {config.allow_unrestricted_mode}")
    logger.info(f"  - Filters: {config.policy_filters_enabled}")
    logger.info(f"  - Curriculum: {config.content_moderation}")
    logger.info(f"  - External Calls: {config.allow_external_calls}")


def reset_policy_engine():
    """Reset the global policy engine (for testing)."""
    global _policy_engine
    with _policy_lock:
        _policy_engine = None
