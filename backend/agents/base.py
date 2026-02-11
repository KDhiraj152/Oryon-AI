"""
Base Agent Protocol and Infrastructure
========================================

Defines the contract all agents must implement,
the message types for inter-agent communication,
and the registry for agent lifecycle management.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AgentStatus(str, Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class MessageType(str, Enum):
    """Standard message types for inter-agent communication."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    METRIC = "metric"
    COMMAND = "command"
    ERROR = "error"


@dataclass
class AgentMessage:
    """
    Typed message for inter-agent communication.

    Immutable after creation — agents should not modify received messages.
    """
    msg_type: MessageType
    sender: str
    recipient: str  # Agent name or "*" for broadcast
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more urgent

    def reply(self, payload: dict[str, Any], msg_type: MessageType = MessageType.RESPONSE) -> "AgentMessage":
        """Create a reply message to this message."""
        return AgentMessage(
            msg_type=msg_type,
            sender=self.recipient,
            recipient=self.sender,
            payload=payload,
            correlation_id=self.correlation_id,
        )


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Contract:
    - Agents are stateless across requests (state is in external stores)
    - Each agent has a mailbox (async queue) for incoming messages
    - Agents process messages sequentially from their mailbox
    - No shared mutable state — communicate only via messages
    """

    def __init__(self, name: str, max_mailbox_size: int = 1000):
        self.name = name
        self.status = AgentStatus.IDLE
        self._mailbox: asyncio.Queue[AgentMessage] = asyncio.Queue(maxsize=max_mailbox_size)
        self._subscribers: dict[str, list[Callable]] = {}
        self._metrics: deque = deque(maxlen=10000)
        self._started_at: float = 0
        self._processed_count: int = 0
        self._error_count: int = 0
        self._task: asyncio.Task | None = None

    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """
        Process a single message. This is the core agent logic.

        Returns:
            Optional reply message, or None if no reply needed.
        """
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """One-time initialization (load configs, connect to services)."""
        ...

    async def start(self) -> None:
        """Start the agent's message processing loop."""
        await self.initialize()
        self.status = AgentStatus.RUNNING
        self._started_at = time.time()
        self._task = asyncio.create_task(self._run_loop(), name=f"agent-{self.name}")
        logger.info(f"Agent '{self.name}' started")

    async def stop(self) -> None:
        """Gracefully stop the agent."""
        self.status = AgentStatus.STOPPED
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:  # NOSONAR
                pass  # Expected after task.cancel() during shutdown
        logger.info(f"Agent '{self.name}' stopped (processed={self._processed_count}, errors={self._error_count})")

    async def send(self, message: AgentMessage) -> None:
        """Send a message to this agent's mailbox."""
        try:
            self._mailbox.put_nowait(message)
            await asyncio.sleep(0)  # yield to event loop
        except asyncio.QueueFull:
            logger.warning(f"Agent '{self.name}' mailbox full, dropping message {message.correlation_id}")

    async def _run_loop(self) -> None:
        """Main message processing loop."""
        while self.status == AgentStatus.RUNNING:
            try:
                message = await asyncio.wait_for(self._mailbox.get(), timeout=1.0)
                start = time.perf_counter()

                try:
                    reply = await self.handle_message(message)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self._processed_count += 1

                    # Record metric
                    self._metrics.append({
                        "type": message.msg_type.value,
                        "latency_ms": elapsed_ms,
                        "timestamp": time.time(),
                    })

                    # Route reply if needed
                    if reply:
                        await self._route_reply(reply)

                except Exception as e:
                    self._error_count += 1
                    logger.error(f"Agent '{self.name}' error processing message: {e}", exc_info=True)

            except asyncio.TimeoutError:
                continue  # Just check status and loop
            except asyncio.CancelledError:
                logger.debug(f"Agent '{self.name}' loop cancelled")
                raise  # re-raise CancelledError per sonar

    async def _route_reply(self, reply: AgentMessage) -> None:
        """Route a reply to the appropriate agent via registry."""
        registry = AgentRegistry.get_instance()
        if registry:
            await registry.route_message(reply)

    def get_stats(self) -> dict[str, Any]:
        """Get agent performance statistics."""
        uptime = time.time() - self._started_at if self._started_at > 0 else 0
        recent = list(self._metrics)[-100:]
        avg_latency = sum(m["latency_ms"] for m in recent) / max(1, len(recent))

        return {
            "name": self.name,
            "status": self.status.value,
            "uptime_s": round(uptime, 1),
            "processed": self._processed_count,
            "errors": self._error_count,
            "mailbox_size": self._mailbox.qsize(),
            "avg_latency_ms": round(avg_latency, 2),
            "error_rate": round(self._error_count / max(1, self._processed_count), 4),
        }


class AgentRegistry:
    """
    Registry for all active agents. Handles message routing.

    Singleton — AgentRegistry() always returns the same instance.
    """

    _instance: Optional["AgentRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once; subsequent AgentRegistry() calls return
        # the existing instance without resetting state.
        if not hasattr(self, "_initialized"):
            self._agents: dict[str, BaseAgent] = {}
            self._message_log: deque = deque(maxlen=1000)
            self._initialized = True

    @classmethod
    def get_instance(cls) -> Optional["AgentRegistry"]:
        """Get the singleton registry instance (None if not yet created)."""
        return cls._instance

    @classmethod
    def create(cls) -> "AgentRegistry":
        """Create and set the singleton registry."""
        return cls()

    @classmethod
    def reset(cls):
        """Reset the registry (for testing)."""
        if cls._instance is not None and hasattr(cls._instance, "_initialized"):
            del cls._instance._initialized
        cls._instance = None

    def register(self, agent: BaseAgent) -> None:
        """Register an agent."""
        self._agents[agent.name] = agent
        logger.info(f"Agent '{agent.name}' registered")

    def unregister(self, name: str) -> None:
        """Unregister an agent."""
        self._agents.pop(name, None)

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get an agent by name."""
        return self._agents.get(name)

    async def route_message(self, message: AgentMessage) -> None:
        """Route a message to the target agent(s)."""
        self._message_log.append({
            "from": message.sender,
            "to": message.recipient,
            "type": message.msg_type.value,
            "correlation_id": message.correlation_id,
            "timestamp": message.timestamp,
        })

        if message.recipient == "*":
            # Broadcast to all agents except sender
            for name, agent in self._agents.items():
                if name != message.sender:
                    await agent.send(message)
        else:
            target_agent = self._agents.get(message.recipient)
            if target_agent:
                await target_agent.send(message)
            else:
                logger.warning(f"No agent '{message.recipient}' found for message from '{message.sender}'")

    async def start_all(self) -> None:
        """Start all registered agents."""
        for agent in self._agents.values():
            await agent.start()

    async def stop_all(self) -> None:
        """Stop all registered agents."""
        for agent in self._agents.values():
            await agent.stop()

    def get_all_stats(self) -> dict[str, Any]:
        """Get stats for all agents."""
        return {
            name: agent.get_stats()
            for name, agent in self._agents.items()
        }

    @property
    def agents(self) -> dict[str, BaseAgent]:
        return self._agents
