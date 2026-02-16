"""
Oryon Multi-Agent System
================================

Modular agent framework for self-optimizing AI pipeline.

Agents:
- OrchestratorAgent: Routes requests, coordinates agents
- ModelExecutionAgent: Manages ML model lifecycle and inference
- HardwareOptimizerAgent: Dynamic hardware tuning
- EvaluationAgent: Quality and performance measurement
- ResourceMonitorAgent: Memory, GPU, latency tracking
- SelfImprovementAgent: Closed-loop optimization

Design Principles:
- No shared mutable global state between agents
- Communication via typed async channels (AgentMessage)
- Each agent has clear input/output contracts
- Hot-reload support via agent registry
- All agents implement BaseAgent protocol
"""

from .base import AgentMessage, AgentRegistry, AgentStatus, BaseAgent
from .evaluation import EvaluationAgent
from .hardware_optimizer import HardwareOptimizerAgent
from .model_execution import ModelExecutionAgent
from .orchestrator import OrchestratorAgent
from .resource_monitor import ResourceMonitorAgent
from .self_improvement import SelfImprovementAgent

__all__ = [
    "AgentMessage",
    "AgentRegistry",
    "AgentStatus",
    "BaseAgent",
    "EvaluationAgent",
    "HardwareOptimizerAgent",
    "ModelExecutionAgent",
    "OrchestratorAgent",
    "ResourceMonitorAgent",
    "SelfImprovementAgent",
]
