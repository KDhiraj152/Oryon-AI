"""
Context Management for AI Conversations
========================================

Implements session context with:
- Multi-turn coherence via short-term memory
- Context window management with summarization
- User-approved long-term memory persistence
- Chain-of-thought compression for context overflow
- **Adaptive Context Allocator (NEW)**:
  - Dynamic context window sizing based on query complexity
  - Priority-based context retention (keep relevant, drop irrelevant)
  - Cross-request context pooling for similar queries
"""

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum, StrEnum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

class ContextRole(StrEnum):
    """Message roles in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

# ============================================================================
# QUERY COMPLEXITY ANALYZER
# ============================================================================

class QueryComplexity(StrEnum):
    """Query complexity levels."""

    SIMPLE = "simple"  # Direct Q&A
    MODERATE = "moderate"  # Some reasoning needed
    COMPLEX = "complex"  # Multi-step reasoning
    DEEP = "deep"  # Research-level

class QueryComplexityAnalyzer:
    """
    Analyzes query complexity to determine optimal context allocation.
    """

    COMPLEXITY_SIGNALS: ClassVar[dict] = {
        QueryComplexity.SIMPLE: {
            "patterns": ["what is", "define", "who is", "when"],
            "max_tokens": 2048,
            "context_retention": 0.3,  # Keep 30% of context
        },
        QueryComplexity.MODERATE: {
            "patterns": ["explain", "how does", "why", "compare"],
            "max_tokens": 4096,
            "context_retention": 0.5,
        },
        QueryComplexity.COMPLEX: {
            "patterns": ["analyze", "design", "implement", "step by step"],
            "max_tokens": 6144,
            "context_retention": 0.7,
        },
        QueryComplexity.DEEP: {
            "patterns": ["research", "comprehensive", "detailed analysis", "prove"],
            "max_tokens": 8192,
            "context_retention": 0.9,
        },
    }

    def analyze(self, query: str) -> tuple[QueryComplexity, dict[str, Any]]:
        """
        Analyze query complexity.

        Returns:
            (complexity_level, config_dict)
        """
        query_lower = query.lower()

        # Score each complexity level
        scores: dict[QueryComplexity, float] = {}
        for level, config in self.COMPLEXITY_SIGNALS.items():
            patterns: list[str] = config["patterns"]  # type: ignore[assignment]
            score = sum(1 for p in patterns if p in query_lower)
            # Boost by query length (longer = likely more complex)
            length_boost = min(0.5, len(query) / 500)
            scores[level] = score + length_boost

        # Select highest scoring level, default to MODERATE
        best_level = max(scores, key=lambda k: scores[k])
        if scores[best_level] < 0.5:
            best_level = QueryComplexity.MODERATE

        config = self.COMPLEXITY_SIGNALS[best_level]
        return best_level, {
            "max_tokens": config["max_tokens"],
            "context_retention": config["context_retention"],
        }

# ============================================================================
# ADAPTIVE CONTEXT ALLOCATOR
# ============================================================================

@dataclass
class ContextAllocation:
    """Allocation decision for a context."""

    max_tokens: int
    retention_ratio: float
    priority_messages: list[int]  # Indices of priority messages
    should_summarize: bool
    estimated_quality: float

    @property
    def total_tokens(self) -> int:
        """Alias for max_tokens for compatibility."""
        return self.max_tokens

    @property
    def priority(self) -> float:
        """Get priority level based on retention ratio and quality."""
        return self.retention_ratio * self.estimated_quality

class AdaptiveContextAllocator:
    """
    Dynamically allocates context window based on query needs.

    Features:
    - Complexity-aware token budgeting
    - Relevance-based message prioritization
    - Cross-query context pooling
    - Memory pressure awareness
    """

    # Base allocations by complexity
    BASE_ALLOCATIONS: ClassVar[dict] = {
        QueryComplexity.SIMPLE: 2048,
        QueryComplexity.MODERATE: 4096,
        QueryComplexity.COMPLEX: 6144,
        QueryComplexity.DEEP: 8192,
    }

    def __init__(self, max_total_tokens: int = 8192):
        self.max_total_tokens = max_total_tokens
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self._relevance_cache: dict[str, float] = {}
        self._lock = threading.Lock()

    def allocate(
        self,
        query: str,
        messages: list["Message"] | None = None,
        available_memory_mb: float = 1000.0,
        intent: str | None = None,
    ) -> ContextAllocation:
        """
        Allocate context window for a query.

        Args:
            query: The incoming query
            messages: Existing conversation messages (optional)
            available_memory_mb: Available memory for context
            intent: Optional intent hint for allocation

        Returns:
            ContextAllocation with optimal settings
        """
        # Handle case where messages is actually an intent string (backwards compat)
        if isinstance(messages, str):
            intent = messages
            messages = None

        messages = messages or []

        # Analyze query complexity
        complexity, config = self.complexity_analyzer.analyze(query)

        # Adjust based on intent if provided
        if intent:
            intent_boosts = {
                "explanation": 1.3,
                "question": 1.0,
                "search": 1.2,
                "code": 1.4,
                "creative": 1.5,
                "conversational": 0.8,
            }
            intent_boost = intent_boosts.get(intent.lower(), 1.0)
        else:
            intent_boost = 1.0

        # Base allocation from complexity
        base_tokens = self.BASE_ALLOCATIONS.get(complexity, 4096)
        base_tokens = int(base_tokens * intent_boost)

        # Adjust for memory pressure (assume ~0.5KB per token)
        memory_tokens = int(available_memory_mb * 2000)  # 2000 tokens per MB
        max_tokens = min(base_tokens, memory_tokens, self.max_total_tokens)

        # Calculate message relevance and priority (if messages provided)
        priority_indices = (
            self._prioritize_messages(query, messages) if messages else []
        )

        # Determine if summarization is needed
        total_message_tokens = sum(
            getattr(m, "token_count", len(str(m)) // 4) for m in messages
        )
        should_summarize = total_message_tokens > max_tokens * 0.8

        # Estimate quality (rough heuristic)
        if should_summarize:
            estimated_quality = 0.8  # Summarization loses some detail
        elif total_message_tokens < max_tokens * 0.5:
            estimated_quality = 1.0  # Full context available
        else:
            estimated_quality = 0.9

        return ContextAllocation(
            max_tokens=max_tokens,
            retention_ratio=config["context_retention"],
            priority_messages=priority_indices[:10],  # Top 10 priority
            should_summarize=should_summarize,
            estimated_quality=estimated_quality,
        )

    def _prioritize_messages(self, query: str, messages: list["Message"]) -> list[int]:
        """
        Prioritize messages by relevance to current query.

        Returns indices sorted by relevance (highest first).
        """
        if not messages:
            return []

        query_words = set(query.lower().split())
        scored = []

        for i, msg in enumerate(messages):
            # Recency boost (more recent = higher priority)
            recency_score = i / len(messages)

            # Get message content (handle both Message objects and strings)
            content = msg.content if hasattr(msg, "content") else str(msg)

            # Relevance score (word overlap)
            msg_words = set(content.lower().split())
            overlap = len(query_words & msg_words) / max(len(query_words), 1)

            # Role boost (user questions and assistant answers are important)
            role = getattr(msg, "role", None)
            role_boost = 0.2 if role in (ContextRole.USER, ContextRole.ASSISTANT) else 0

            total_score = (recency_score * 0.4) + (overlap * 0.4) + role_boost
            scored.append((i, total_score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored]

    def apply_allocation(
        self,
        allocation: ContextAllocation,
        messages: list["Message"],
    ) -> list["Message"]:
        """
        Apply allocation to select which messages to keep.

        Returns filtered message list.
        """
        if not messages:
            return []

        # Always keep most recent messages
        recent_count = max(2, int(len(messages) * allocation.retention_ratio))
        recent_messages = messages[-recent_count:]

        # Add priority messages not in recent
        priority_set = set(allocation.priority_messages)
        for idx in allocation.priority_messages:
            if idx < len(messages) - recent_count:
                # This is a priority message not in recent
                recent_messages.insert(0, messages[idx])

        # Check token budget
        total_tokens = sum(m.token_count for m in recent_messages)
        while total_tokens > allocation.max_tokens and len(recent_messages) > 2:
            # Remove oldest non-priority message
            for i, msg in enumerate(recent_messages):
                orig_idx = messages.index(msg) if msg in messages else -1
                if orig_idx not in priority_set:
                    removed = recent_messages.pop(i)
                    total_tokens -= removed.token_count
                    break
            else:
                # All are priority, remove oldest anyway
                removed = recent_messages.pop(0)
                total_tokens -= removed.token_count

        return recent_messages

# ============================================================================
# CROSS-QUERY CONTEXT POOL
# ============================================================================

@dataclass
class PooledContext:
    """Context stored in the pool."""

    query_hash: str
    query_embedding: list[float] | None
    messages: list[dict[str, Any]]
    created_at: datetime
    access_count: int = 0

class ContextPool:
    """
    Pools context across similar queries for reuse.

    When a user asks a similar question to a previous one,
    we can leverage the previous context for faster, better responses.
    """

    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.8):
        self._pool: dict[str, PooledContext] = {}
        self._max_size = max_size
        self._similarity_threshold = similarity_threshold
        self._lock = threading.Lock()

    @property
    def max_size(self) -> int:
        """Maximum pool size."""
        return self._max_size

    @property
    def current_size(self) -> int:
        """Current number of entries in pool."""
        with self._lock:
            return len(self._pool)

    def store(
        self,
        query: str,
        messages: list["Message"],
        query_embedding: list[float] | None = None,
    ) -> str:
        """Store context in pool. Returns query hash."""
        query_hash = hashlib.md5(
            query.lower().strip().encode(), usedforsecurity=False
        ).hexdigest()[:12]

        with self._lock:
            # Evict if at capacity
            if len(self._pool) >= self._max_size:
                self._evict_lru()

            self._pool[query_hash] = PooledContext(
                query_hash=query_hash,
                query_embedding=query_embedding,
                messages=[m.to_dict() for m in messages],
                created_at=datetime.now(UTC),
            )

        return query_hash

    def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Retrieve similar context from pool.

        Returns messages if similar query found, else None.
        """
        query_hash = hashlib.md5(
            query.lower().strip().encode(), usedforsecurity=False
        ).hexdigest()[:12]

        with self._lock:
            # Exact match
            if query_hash in self._pool:
                ctx = self._pool[query_hash]
                ctx.access_count += 1
                return ctx.messages

            # TODO: Semantic similarity search using embeddings
            # For now, just return None for non-exact matches

        return None

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._pool:
            return

        # Find entry with lowest access count and oldest
        lru_key = min(
            self._pool.keys(),
            key=lambda k: (self._pool[k].access_count, self._pool[k].created_at),
        )
        del self._pool[lru_key]

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "size": len(self._pool),
                "max_size": self._max_size,
                "total_accesses": sum(c.access_count for c in self._pool.values()),
            }

@dataclass
class Message:
    """Single message in conversation."""

    role: ContextRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=ContextRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.now(UTC).isoformat())
            ),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0),
        )

@dataclass
class ConversationContext:
    """
    Manages conversation context for a single session.

    Features:
    - Token-aware context window management
    - Automatic summarization when context exceeds window
    - Relevance scoring for message retention
    - Chain-of-thought compression
    """

    conversation_id: str
    user_id: str | None = None
    max_tokens: int = 8192  # Context window size
    messages: list[Message] = field(default_factory=list)
    summary: str | None = None
    total_tokens: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Long-term memory (user-approved only)
    long_term_memory: list[str] = field(default_factory=list)
    memory_consent: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(
        self,
        role: ContextRole,
        content: str,
        metadata: dict[str, Any] | None = None,
        token_count: int | None = None,
    ) -> Message:
        """Add a message to the conversation."""
        # Estimate tokens if not provided (rough: 4 chars per token)
        if token_count is None:
            token_count = len(content) // 4 + 1

        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            token_count=token_count,
        )

        self.messages.append(message)
        self.total_tokens += token_count
        self.updated_at = datetime.now(UTC)

        # Check if we need to compress context
        if self.total_tokens > self.max_tokens * 0.9:
            self._compress_context()

        return message

    def get_context_messages(
        self, max_messages: int | None = None
    ) -> list[dict[str, str]]:
        """
        Get messages formatted for LLM input.

        Returns messages with summary prepended if context was compressed.
        """
        result = []

        # Add summary as system message if we have one
        if self.summary:
            result.append(
                {
                    "role": "system",
                    "content": f"[Previous conversation summary]: {self.summary}",
                }
            )

        # Get messages (optionally limited)
        messages = self.messages
        if max_messages and len(messages) > max_messages:
            messages = messages[-max_messages:]

        for msg in messages:
            result.append(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                }
            )

        return result

    def _compress_context(self):
        """
        Compress context when approaching token limit.

        Strategy:
        1. Summarize older messages
        2. Keep recent messages (last 20% of context)
        3. Store summary for context continuity
        """
        if len(self.messages) < 4:
            return

        # Determine split point (keep last 20% of tokens)
        target_keep_tokens = int(self.max_tokens * 0.2)

        keep_messages = []
        keep_tokens = 0

        # Work backwards to find messages to keep
        for msg in reversed(self.messages):
            if keep_tokens + msg.token_count > target_keep_tokens:
                break
            keep_messages.insert(0, msg)
            keep_tokens += msg.token_count

        # Messages to summarize (everything before keep_messages)
        summarize_count = len(self.messages) - len(keep_messages)
        if summarize_count < 2:
            return

        to_summarize = self.messages[:summarize_count]

        # Create summary (synchronous for simplicity - async in production)
        summary_text = self._create_summary(to_summarize)

        # Update state
        self.summary = summary_text
        self.messages = keep_messages
        self.total_tokens = keep_tokens + (len(summary_text) // 4)

        logger.info(
            f"Context compressed: {summarize_count} messages → summary, "
            f"keeping {len(keep_messages)} recent messages"
        )

    def _create_summary(self, messages: list[Message]) -> str:
        """
        Create a concise summary of messages.

        For production, this should use the LLM. Here we use a simple extraction.
        """
        key_points = []

        for msg in messages:
            if msg.role == ContextRole.USER:
                # Extract user's main question/intent
                content = msg.content.strip()
                if len(content) > 100:
                    content = content[:100] + "..."
                key_points.append(f"User asked: {content}")
            elif msg.role == ContextRole.ASSISTANT:
                # Extract first sentence of response
                content = msg.content.strip()
                first_sentence = (
                    content.split(".")[0] if "." in content else content[:100]
                )
                key_points.append(f"Assistant: {first_sentence}")

        return " | ".join(key_points[-5:])  # Keep last 5 key points

    def approve_long_term_memory(self, content: str) -> None:
        """Store user-approved content in long-term memory."""
        if not self.memory_consent:
            logger.warning("Long-term memory not consented - ignoring store request")
            return

        self.long_term_memory.append(content)
        logger.info(
            f"Stored long-term memory item for conversation {self.conversation_id}"
        )

    def get_long_term_context(self) -> list[str]:
        """Get relevant long-term memory items."""
        return self.long_term_memory.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "max_tokens": self.max_tokens,
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "long_term_memory": self.long_term_memory,
            "memory_consent": self.memory_consent,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationContext":
        """Deserialize context from dictionary."""
        ctx = cls(
            conversation_id=data["conversation_id"],
            user_id=data.get("user_id"),
            max_tokens=data.get("max_tokens", 8192),
            summary=data.get("summary"),
            total_tokens=data.get("total_tokens", 0),
            long_term_memory=data.get("long_term_memory", []),
            memory_consent=data.get("memory_consent", False),
            metadata=data.get("metadata", {}),
        )
        ctx.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        ctx.created_at = datetime.fromisoformat(
            data.get("created_at", datetime.now(UTC).isoformat())
        )
        ctx.updated_at = datetime.fromisoformat(
            data.get("updated_at", datetime.now(UTC).isoformat())
        )
        return ctx

class ContextManager:
    """
    Manages multiple conversation contexts with caching.

    Features:
    - Redis-backed persistence for production
    - In-memory cache for fast access
    - Automatic expiration of stale contexts
    - Thread-safe operations
    """

    def __init__(self, redis_client=None, ttl_hours: int = 24):
        self._contexts: dict[str, ConversationContext] = {}
        self._redis = redis_client
        self._ttl_hours = ttl_hours
        self._cache_key_prefix = "ai_context:"

    def get_or_create(
        self,
        conversation_id: str,
        user_id: str | None = None,
        max_tokens: int = 8192,
    ) -> ConversationContext:
        """Get existing context or create new one."""
        # Check in-memory cache first
        if conversation_id in self._contexts:
            return self._contexts[conversation_id]

        # Try Redis if available
        if self._redis:
            cache_key = f"{self._cache_key_prefix}{conversation_id}"
            try:
                cached = self._redis.get(cache_key)
                if cached:
                    import json

                    ctx = ConversationContext.from_dict(json.loads(cached))
                    self._contexts[conversation_id] = ctx
                    return ctx
            except (OSError, ConnectionError, TimeoutError) as e:
                logger.warning("Redis get failed: %s", e)

        # Create new context
        ctx = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            max_tokens=max_tokens,
        )
        self._contexts[conversation_id] = ctx
        return ctx

    def save(self, context: ConversationContext) -> None:
        """Save context to cache and persistent storage."""
        self._contexts[context.conversation_id] = context

        if self._redis:
            cache_key = f"{self._cache_key_prefix}{context.conversation_id}"
            try:
                import json

                self._redis.setex(
                    cache_key, self._ttl_hours * 3600, json.dumps(context.to_dict())
                )
            except (OSError, ConnectionError, TimeoutError) as e:
                logger.warning("Redis save failed: %s", e)

    def delete(self, conversation_id: str) -> None:
        """Delete a conversation context."""
        if conversation_id in self._contexts:
            del self._contexts[conversation_id]

        if self._redis:
            cache_key = f"{self._cache_key_prefix}{conversation_id}"
            try:
                self._redis.delete(cache_key)
            except (OSError, ConnectionError, TimeoutError) as e:
                logger.warning("Redis delete failed: %s", e)

    def get_active_count(self) -> int:
        """Get count of active contexts in memory."""
        return len(self._contexts)

    def cleanup_stale(self, max_age_hours: int = 24) -> int:
        """Remove contexts older than max_age_hours. Returns count removed."""
        now = datetime.now(UTC)
        stale_ids = []

        for conv_id, ctx in self._contexts.items():
            age_hours = (now - ctx.updated_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                stale_ids.append(conv_id)

        for conv_id in stale_ids:
            del self._contexts[conv_id]

        return len(stale_ids)
