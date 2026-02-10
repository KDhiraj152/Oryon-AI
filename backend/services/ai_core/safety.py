"""
Safety Guard - Content Filtering and Secret Detection
======================================================

Implements safety features:
- Harmful content detection and filtering
- Secret/credential detection and redaction
- PII (Personal Identifiable Information) handling
- Input sanitization

NOTE: This module now integrates with the centralized PolicyEngine.
When ALLOW_UNRESTRICTED_MODE=true, most filtering is bypassed.
See backend/policy/policy_module.py for configuration.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Import policy module for centralized configuration
try:
    from ...policy import PolicyMode, get_policy_engine

    _POLICY_AVAILABLE = True
except ImportError:
    _POLICY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk levels for detected content."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecretType(str, Enum):
    """Types of secrets that can be detected."""

    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    PRIVATE_KEY = "private_key"
    AWS_KEY = "aws_key"
    DATABASE_URL = "database_url"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    UNKNOWN = "unknown"


@dataclass
class SecretMatch:
    """A detected secret in text."""

    secret_type: SecretType
    value: str
    start: int
    end: int
    risk_level: RiskLevel
    redacted: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.secret_type.value,
            "risk_level": self.risk_level.value,
            "start": self.start,
            "end": self.end,
            "redacted": self.redacted,
        }


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""

    is_safe: bool
    risk_level: RiskLevel
    issues: list[str]
    filtered_content: str | None = None
    secrets_found: list[SecretMatch] | None = None

    def __post_init__(self):
        if self.secrets_found is None:
            self.secrets_found = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "risk_level": self.risk_level.value,
            "issues": self.issues,
            "secrets_count": len(self.secrets_found),
        }


class SecretScanner:
    """
    Scans text for secrets and sensitive information.

    Detects:
    - API keys (AWS, Google, GitHub, etc.)
    - Passwords and tokens
    - Private keys
    - Database connection strings
    - Credit card numbers
    - SSN and other PII
    """

    # Secret detection patterns
    PATTERNS = {
        SecretType.AWS_KEY: [
            (r"AKIA[0-9A-Z]{16}", RiskLevel.CRITICAL),  # AWS Access Key ID
            (
                r'(?i)aws[_-]?secret[_-]?access[_-]?key["\']?\s*[:=]\s*["\']?([A-Za-z0-9/+=]{40})',
                RiskLevel.CRITICAL,
            ),
        ],
        SecretType.API_KEY: [
            (
                r'(?i)(api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})',
                RiskLevel.HIGH,
            ),
            (r"(?i)bearer\s+[A-Za-z0-9_\-\.]+", RiskLevel.HIGH),
            (r"sk-[A-Za-z0-9]{48}", RiskLevel.CRITICAL),  # OpenAI API key
            (
                r"ghp_[A-Za-z0-9]{36}",
                RiskLevel.CRITICAL,
            ),  # GitHub personal access token
            (r"gho_[A-Za-z0-9]{36}", RiskLevel.CRITICAL),  # GitHub OAuth token
        ],
        SecretType.PASSWORD: [
            (
                r'(?i)(password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^\s"\']{8,})',
                RiskLevel.CRITICAL,
            ),
            (r'(?i)secret["\']?\s*[:=]\s*["\']?([^\s"\']{8,})', RiskLevel.HIGH),
        ],
        SecretType.TOKEN: [
            (
                r'(?i)(token|auth_token|access_token)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-\.]{20,})',
                RiskLevel.HIGH,
            ),
            (
                r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*",
                RiskLevel.HIGH,
            ),  # JWT
        ],
        SecretType.PRIVATE_KEY: [
            (
                r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
                RiskLevel.CRITICAL,
            ),
            (r"-----BEGIN PGP PRIVATE KEY BLOCK-----", RiskLevel.CRITICAL),
        ],
        SecretType.DATABASE_URL: [
            (r'(?i)(postgres|mysql|mongodb|redis)://[^\s"\']+', RiskLevel.CRITICAL),
            (r'(?i)database_url["\']?\s*[:=]\s*["\']?([^\s"\']+)', RiskLevel.CRITICAL),
        ],
        SecretType.CREDIT_CARD: [
            (
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9][0-9])[0-9]{12})\b",
                RiskLevel.CRITICAL,
            ),
        ],
        SecretType.SSN: [
            (r"\b\d{3}-\d{2}-\d{4}\b", RiskLevel.CRITICAL),
        ],
        SecretType.EMAIL: [
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", RiskLevel.MEDIUM),
        ],
        SecretType.PHONE: [
            (
                r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                RiskLevel.MEDIUM,
            ),
        ],
    }

    def __init__(self):
        # Compile patterns for efficiency
        self._compiled_patterns: dict[
            SecretType, list[tuple[re.Pattern, RiskLevel]]
        ] = {}
        for secret_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[secret_type] = [
                (re.compile(pattern), risk) for pattern, risk in patterns
            ]

    def scan(self, text: str) -> list[SecretMatch]:
        """Scan text for secrets."""
        matches = []

        for secret_type, patterns in self._compiled_patterns.items():
            for pattern, risk_level in patterns:
                for match in pattern.finditer(text):
                    # Get the matched value
                    value = match.group(0)

                    # Skip if too short (likely false positive)
                    if len(value) < 8 and secret_type not in [SecretType.SSN]:
                        continue

                    matches.append(
                        SecretMatch(
                            secret_type=secret_type,
                            value=value,
                            start=match.start(),
                            end=match.end(),
                            risk_level=risk_level,
                            redacted=self._redact_value(value, secret_type),
                        )
                    )

        return matches

    def redact(self, text: str) -> tuple[str, list[SecretMatch]]:
        """Redact all secrets from text."""
        matches = self.scan(text)

        if not matches:
            return text, []

        # Sort by position (reverse) to avoid offset issues
        matches.sort(key=lambda m: m.start, reverse=True)

        result = text
        for match in matches:
            result = result[: match.start] + match.redacted + result[match.end :]

        return result, matches

    def _redact_value(self, value: str, secret_type: SecretType) -> str:
        """Create redacted version of a secret value."""
        if len(value) <= 8:
            return f"[REDACTED_{secret_type.value.upper()}]"

        # Show first and last few characters
        prefix = value[:4]
        suffix = value[-4:] if len(value) > 8 else ""

        return f"{prefix}...{suffix}[REDACTED]"


class SafetyGuard:
    """
    Guards against harmful content and security issues.

    Features:
    - Input sanitization
    - Harmful content detection
    - Secret redaction
    - PII protection
    - Jailbreak detection
    """

    # Harmful content patterns (simplified - production would use ML models)
    HARMFUL_PATTERNS = [
        # Violence
        (
            r"(?i)\b(how to|ways to|instructions for)\s+(kill|harm|hurt|attack|murder)\b",
            RiskLevel.CRITICAL,
        ),
        (r"(?i)\b(make|build|create)\s+(bomb|explosive|weapon)\b", RiskLevel.CRITICAL),
        # Illegal activities
        (
            r"(?i)\b(hack|crack|steal)\s+(password|account|credit card)\b",
            RiskLevel.HIGH,
        ),
        (r"(?i)\b(create|generate)\s+(malware|virus|ransomware)\b", RiskLevel.CRITICAL),
        # Jailbreak attempts
        (
            r"(?i)(ignore|forget|disregard)\s+(previous|all)\s+(instructions|rules|guidelines)",
            RiskLevel.HIGH,
        ),
        (
            r"(?i)pretend\s+you\s+(are|can|have)\s+no\s+(restrictions|limits|rules)",
            RiskLevel.HIGH,
        ),
        (r"(?i)DAN\s+mode|do\s+anything\s+now", RiskLevel.HIGH),
    ]

    # Allow patterns (educational context)
    EDUCATIONAL_PATTERNS = [
        r"(?i)(explain|understand|learn about|history of|science of)",
        r"(?i)(for educational purposes|in theory|academically)",
        r"(?i)(what is|how does|why does)",
    ]

    def __init__(self):
        self.secret_scanner = SecretScanner()

        # Check if we should use policy engine
        self._use_policy_engine = _POLICY_AVAILABLE
        self._policy_engine = None

        # Compile patterns (kept for fallback/legacy compatibility)
        self._harmful_patterns = [
            (re.compile(pattern), risk) for pattern, risk in self.HARMFUL_PATTERNS
        ]
        self._educational_patterns = [
            re.compile(pattern) for pattern in self.EDUCATIONAL_PATTERNS
        ]

    def _get_policy_engine(self):
        """Get the policy engine instance."""
        if self._policy_engine is None and self._use_policy_engine:
            try:
                self._policy_engine = get_policy_engine()
            except Exception as e:
                logger.warning(f"Could not get policy engine: {e}")
                self._use_policy_engine = False
        return self._policy_engine

    def check_input(self, text: str) -> SafetyCheckResult:
        """Check input text for safety issues.

        NOTE: In UNRESTRICTED mode (via policy engine), most checks are bypassed.
        """
        # Try to use policy engine first
        policy = self._get_policy_engine()
        if policy and policy.mode == PolicyMode.UNRESTRICTED:
            # In unrestricted mode, still do secret scanning but skip content checks
            secrets = (
                self.secret_scanner.scan(text) if policy.config.redact_secrets else []
            )
            return SafetyCheckResult(
                is_safe=True,
                risk_level=RiskLevel.LOW,
                issues=[],
                secrets_found=secrets,
            )

        issues = []
        max_risk = RiskLevel.LOW

        # Check for harmful content
        for pattern, risk in self._harmful_patterns:
            if pattern.search(text):
                # Check if it's in educational context
                is_educational = any(
                    ep.search(text) for ep in self._educational_patterns
                )

                if not is_educational:
                    issues.append("Potentially harmful content detected")
                    if risk.value > max_risk.value:
                        max_risk = risk

        # Check for secrets in input
        secrets = self.secret_scanner.scan(text)
        if secrets:
            critical_secrets = [
                s for s in secrets if s.risk_level == RiskLevel.CRITICAL
            ]
            if critical_secrets:
                issues.append(
                    f"Critical secrets detected in input ({len(critical_secrets)} found)"
                )
                max_risk = RiskLevel.CRITICAL

        return SafetyCheckResult(
            is_safe=max_risk in [RiskLevel.LOW, RiskLevel.MEDIUM],
            risk_level=max_risk,
            issues=issues,
            secrets_found=secrets,
        )

    def filter_response(self, text: str) -> str:
        """Filter harmful content from response.

        NOTE: In UNRESTRICTED mode, returns text unchanged.
        """
        # Check policy engine mode
        policy = self._get_policy_engine()
        if policy and policy.mode == PolicyMode.UNRESTRICTED:
            # In unrestricted mode, pass through unchanged
            return text

        # This is a simple filter - production would use ML models
        # Remove any code that looks like it could be malicious
        filtered = re.sub(
            r"(?i)(os\.system|subprocess|eval|exec)\s*\([^)]+\)",
            "[CODE_REMOVED_FOR_SAFETY]",
            text,
        )

        return filtered

    def redact_secrets(self, text: str) -> str:
        """Redact secrets from text."""
        # Check if redaction is enabled via policy
        policy = self._get_policy_engine()
        if policy and not policy.config.redact_secrets:
            return text

        redacted, _ = self.secret_scanner.redact(text)
        return redacted

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input."""
        # Remove null bytes
        text = text.replace("\x00", "")

        # Limit length
        if len(text) > 50000:
            text = text[:50000] + "\n\n[Content truncated for safety]"

        # Remove control characters (except newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text

    def check_and_process(self, text: str) -> tuple[str, SafetyCheckResult]:
        """
        Full safety processing: check, sanitize, and redact.

        NOTE: In UNRESTRICTED mode, only basic sanitization is performed.

        Returns:
            Tuple of (processed_text, safety_result)
        """
        # Sanitize first
        sanitized = self.sanitize_input(text)

        # Check for issues
        result = self.check_input(sanitized)

        # Redact secrets if found
        if result.secrets_found:
            sanitized = self.redact_secrets(sanitized)

        result.filtered_content = sanitized

        return sanitized, result

    def get_rejection_message(self, result: SafetyCheckResult) -> str:
        """Get appropriate rejection message for unsafe content."""
        if result.risk_level == RiskLevel.CRITICAL:
            return (
                "I'm sorry, but I can't help with that request as it may involve "
                "harmful or dangerous content. If you're researching a topic for "
                "educational purposes, please rephrase your question to clarify the context."
            )
        elif result.risk_level == RiskLevel.HIGH:
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
