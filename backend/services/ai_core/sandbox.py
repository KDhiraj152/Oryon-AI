"""
Sandboxed Tool Execution for AI Assistant.

Provides safe, isolated execution of tools including:
- Code execution (Python, JavaScript)
- Math calculations
- Web search (when available)
- File operations (read-only)

Security-first design with resource limits and permission controls.
"""

import ast
import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ToolPermission(str, Enum):
    """Permission levels for tools."""

    READ_ONLY = "read_only"  # Can read but not modify anything
    COMPUTE = "compute"  # Can perform calculations
    NETWORK = "network"  # Can make network requests
    FILE_WRITE = "file_write"  # Can write files (restricted paths)
    FULL = "full"  # All permissions (admin only)


@dataclass
class ExecutionResult:
    """Result of tool execution."""

    success: bool
    output: Any
    error: str | None = None
    stderr: str | None = None
    execution_time_ms: float = 0.0
    memory_used_bytes: int = 0
    truncated: bool = False
    sandbox_id: str = ""


@dataclass
class ResourceLimits:
    """Resource limits for sandboxed execution."""

    max_execution_time_seconds: float = 5.0
    max_memory_mb: int = 128
    max_output_size_bytes: int = 65536  # 64KB
    max_file_size_bytes: int = 1048576  # 1MB
    max_network_requests: int = 5
    allowed_paths: list[str] = field(default_factory=list)


class SandboxedTool(ABC):
    """Base class for sandboxed tools."""

    name: str
    description: str
    required_permissions: list[ToolPermission]

    @abstractmethod
    async def execute(
        self, params: dict[str, Any], limits: ResourceLimits
    ) -> ExecutionResult:
        """Execute the tool with given parameters."""
        pass


class PythonCodeRunner(SandboxedTool):
    """
    Sandboxed Python code execution.

    Uses restricted builtins and timeout enforcement.
    """

    name = "python_execute"
    description = "Execute Python code in a sandboxed environment"
    required_permissions = [ToolPermission.COMPUTE]

    # Safe builtins whitelist
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
    }

    # Blocked patterns in code
    BLOCKED_PATTERNS = [
        "import os",
        "import sys",
        "import subprocess",
        "__import__",
        "eval(",
        "exec(",
        "compile(",
        "open(",
        "file(",
        "input(",
        "__builtins__",
        "__code__",
        "__globals__",
        "os.system",
        "subprocess.",
        "socket.",
        "pickle.",
        "marshal.",
        "ctypes.",
    ]

    # Dangerous attribute names that could be used for sandbox escape
    BLOCKED_ATTRIBUTES = {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__dict__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
        "__init_subclass__",
        "__class_getitem__",
        "func_globals",
        "func_code",
        "gi_frame",
        "gi_code",
        "co_code",
        "f_globals",
        "f_locals",
        "f_builtins",
    }

    def __init__(self):
        from ...core.optimized.thread_pool_manager import get_io_executor
        self._executor = get_io_executor()  # Shared I/O pool

    def _validate_code(self, code: str) -> str | None:
        """Validate code for security issues."""
        code_lower = code.lower()

        # Check for blocked patterns
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                return f"Blocked pattern detected: {pattern}"

        # Try to parse the AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        # Check for dangerous AST nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ["os", "sys", "subprocess", "socket", "pickle"]:
                        return f"Import of '{alias.name}' is not allowed"
            elif isinstance(node, ast.ImportFrom):
                if node.module in ["os", "sys", "subprocess", "socket", "pickle"]:
                    return f"Import from '{node.module}' is not allowed"
            # Block dangerous attribute access patterns (sandbox escape vectors)
            elif isinstance(node, ast.Attribute):
                if node.attr in self.BLOCKED_ATTRIBUTES:
                    return (
                        f"Access to '{node.attr}' is not allowed (security restriction)"
                    )
            # Block subscript access to __class__ etc via strings
            elif isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Constant) and isinstance(
                    node.slice.value, str
                ):
                    if node.slice.value in self.BLOCKED_ATTRIBUTES:
                        return f"Access to '{node.slice.value}' is not allowed (security restriction)"

        return None

    async def execute(
        self, params: dict[str, Any], limits: ResourceLimits
    ) -> ExecutionResult:
        """Execute Python code safely."""
        start_time = datetime.now()
        code = params.get("code", "")

        # Validate code
        validation_error = self._validate_code(code)
        if validation_error:
            return ExecutionResult(
                success=False,
                output=None,
                error=validation_error,
                execution_time_ms=0,
                sandbox_id=self._generate_sandbox_id(code),
            )

        # Create restricted globals
        # Handle __builtins__ being either dict or module
        builtins_dict = (
            __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
        )
        safe_builtins = {
            name: builtins_dict.get(name)
            for name in self.SAFE_BUILTINS
            if name in builtins_dict
        }

        # Add math module (safe)
        import math

        restricted_globals = {
            "__builtins__": safe_builtins,
            "math": math,
            "__name__": "__sandbox__",
        }

        # Capture output
        output_capture: list[str] = []

        def safe_print(*args, **kwargs):
            output_capture.append(" ".join(str(a) for a in args))

        restricted_globals["print"] = safe_print

        try:
            # Run in thread with timeout
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor, self._execute_code, code, restricted_globals, limits
                ),
                timeout=limits.max_execution_time_seconds,
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Combine output
            final_output = "\n".join(output_capture)
            if result is not None:
                final_output = (
                    f"{final_output}\n{result}" if final_output else str(result)
                )

            # Truncate if needed
            truncated = False
            if len(final_output) > limits.max_output_size_bytes:
                final_output = (
                    final_output[: limits.max_output_size_bytes] + "\n... (truncated)"
                )
                truncated = True

            return ExecutionResult(
                success=True,
                output=final_output,
                execution_time_ms=execution_time,
                truncated=truncated,
                sandbox_id=self._generate_sandbox_id(code),
            )

        except TimeoutError:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Execution timeout ({limits.max_execution_time_seconds}s exceeded)",
                execution_time_ms=limits.max_execution_time_seconds * 1000,
                sandbox_id=self._generate_sandbox_id(code),
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return ExecutionResult(
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=execution_time,
                sandbox_id=self._generate_sandbox_id(code),
            )

    def _execute_code(
        self, code: str, restricted_globals: dict, limits: ResourceLimits
    ) -> Any:
        """Execute code in restricted environment."""
        # Compile first to catch syntax errors
        compiled = compile(code, "<sandbox>", "exec")

        # Execute - use same dict for globals and locals so functions are accessible
        # This is necessary for recursive functions and proper scoping
        exec(compiled, restricted_globals)

        # Return last expression value if available
        return restricted_globals.get("result", restricted_globals.get("_"))

    def _generate_sandbox_id(self, code: str) -> str:
        """Generate unique sandbox execution ID."""
        timestamp = datetime.now().isoformat()
        content = f"{code}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class MathCalculator(SandboxedTool):
    """Safe mathematical expression evaluator using AST interpretation (no eval)."""

    name = "calculate"
    description = "Evaluate mathematical expressions safely"
    required_permissions = [ToolPermission.COMPUTE]

    ALLOWED_NAMES = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "int": int,
        "float": float,
    }

    # Allowed binary operators
    BINARY_OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }

    # Allowed unary operators
    UNARY_OPS = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    # Allowed comparison operators
    COMPARE_OPS = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
    }

    def __init__(self):
        import math

        # Add math functions and constants
        self.ALLOWED_NAMES.update(
            {
                name: getattr(math, name)
                for name in dir(math)
                if not name.startswith("_")
            }
        )

    def _safe_eval_node(self, node: ast.AST) -> Any:
        """Safely evaluate an AST node without using eval()."""
        if isinstance(node, ast.Expression):
            return self._safe_eval_node(node.body)
        elif isinstance(node, ast.Constant):
            # Numeric and string constants
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.Name):
            if node.id in self.ALLOWED_NAMES:
                return self.ALLOWED_NAMES[node.id]
            raise ValueError(f"Name '{node.id}' is not allowed")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.BINARY_OPS:
                raise ValueError(f"Operator {op_type.__name__} is not allowed")
            left = self._safe_eval_node(node.left)
            right = self._safe_eval_node(node.right)
            return self.BINARY_OPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.UNARY_OPS:
                raise ValueError(f"Unary operator {op_type.__name__} is not allowed")
            operand = self._safe_eval_node(node.operand)
            return self.UNARY_OPS[op_type](operand)
        elif isinstance(node, ast.Compare):
            # Handle comparison chains like 1 < x < 10
            left = self._safe_eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=False):
                op_type = type(op)
                if op_type not in self.COMPARE_OPS:
                    raise ValueError(f"Comparison {op_type.__name__} is not allowed")
                right = self._safe_eval_node(comparator)
                if not self.COMPARE_OPS[op_type](left, right):
                    return False
                left = right
            return True
        elif isinstance(node, ast.Call):
            # Function calls - only allowed functions
            if isinstance(node.func, ast.Name):
                if node.func.id not in self.ALLOWED_NAMES:
                    raise ValueError(f"Function '{node.func.id}' is not allowed")
                func = self.ALLOWED_NAMES[node.func.id]
                args = [self._safe_eval_node(arg) for arg in node.args]
                return func(*args)
            raise ValueError("Only direct function calls are allowed")
        elif isinstance(node, ast.IfExp):
            # Ternary expression: a if condition else b
            test = self._safe_eval_node(node.test)
            if test:
                return self._safe_eval_node(node.body)
            return self._safe_eval_node(node.orelse)
        elif isinstance(node, ast.List):
            return [self._safe_eval_node(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._safe_eval_node(elt) for elt in node.elts)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    async def execute(
        self, params: dict[str, Any], limits: ResourceLimits
    ) -> ExecutionResult:
        """Evaluate math expression safely using AST interpretation."""
        start_time = datetime.now()
        expression = params.get("expression", "")

        try:
            # Parse expression
            tree = ast.parse(expression, mode="eval")

            # Evaluate using safe AST interpreter (no eval/exec)
            result = self._safe_eval_node(tree)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return ExecutionResult(
                success=True,
                output=result,
                execution_time_ms=execution_time,
                sandbox_id="math",
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return ExecutionResult(
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=execution_time,
                sandbox_id="math",
            )


class JSONProcessor(SandboxedTool):
    """Safe JSON parsing and processing."""

    name = "json_process"
    description = "Parse and process JSON data"
    required_permissions = [ToolPermission.READ_ONLY]

    async def execute(
        self, params: dict[str, Any], limits: ResourceLimits
    ) -> ExecutionResult:
        """Process JSON data."""
        start_time = datetime.now()

        try:
            data = params.get("data", "")
            operation = params.get("operation", "parse")

            if operation == "parse":
                result = json.loads(data) if isinstance(data, str) else data
            elif operation == "stringify":
                result = json.dumps(data, indent=2, default=str)
            elif operation == "keys":
                if isinstance(data, dict):
                    result = list(data.keys())
                else:
                    return ExecutionResult(
                        success=False,
                        output=None,
                        error="Data must be a JSON object to get keys",
                        sandbox_id="json",
                    )
            elif operation == "get":
                path = params.get("path", "")
                result = self._get_by_path(data, path)
            else:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}",
                    sandbox_id="json",
                )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return ExecutionResult(
                success=True,
                output=result,
                execution_time_ms=execution_time,
                sandbox_id="json",
            )

        except json.JSONDecodeError as e:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"JSON parse error: {e}",
                sandbox_id="json",
            )
        except Exception as e:
            return ExecutionResult(
                success=False, output=None, error=str(e), sandbox_id="json"
            )

    def _get_by_path(self, data: Any, path: str) -> Any:
        """Get nested value by dot-notation path."""
        if not path:
            return data

        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list):
                try:
                    idx = int(key)
                    current = current[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current


class ToolSandbox:
    """
    Main sandbox manager for tool execution.

    Coordinates tool registration, permission checking, and execution.
    """

    def __init__(self, default_limits: ResourceLimits | None = None):
        self._tools: dict[str, SandboxedTool] = {}
        self._limits = default_limits or ResourceLimits()
        self._execution_history: list[dict[str, Any]] = []

        # Register built-in tools
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register default sandboxed tools."""
        tools = [
            PythonCodeRunner(),
            MathCalculator(),
            JSONProcessor(),
        ]
        for tool in tools:
            self._tools[tool.name] = tool

    def register_tool(self, tool: SandboxedTool):
        """Register a custom tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools with their metadata."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "permissions": [p.value for p in tool.required_permissions],
            }
            for tool in self._tools.values()
        ]

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        user_permissions: list[ToolPermission] | None = None,
        limits: ResourceLimits | None = None,
    ) -> ExecutionResult:
        """
        Execute a tool with permission checking.

        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            user_permissions: Permissions granted to user
            limits: Resource limits (uses defaults if not provided)

        Returns:
            ExecutionResult with output or error
        """
        if tool_name not in self._tools:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}",
            )

        tool = self._tools[tool_name]

        # Check permissions
        user_perms = user_permissions or [
            ToolPermission.COMPUTE,
            ToolPermission.READ_ONLY,
        ]

        for required in tool.required_permissions:
            if required not in user_perms and ToolPermission.FULL not in user_perms:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Permission denied: {required.value} required for tool '{tool_name}'",
                )

        # Apply limits
        effective_limits = limits or self._limits

        # Execute
        start_time = datetime.now()
        try:
            result = await tool.execute(params, effective_limits)

            # Log execution
            self._execution_history.append(
                {
                    "tool": tool_name,
                    "timestamp": start_time.isoformat(),
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                    "sandbox_id": result.sandbox_id,
                }
            )

            # Trim history
            if len(self._execution_history) > 100:
                self._execution_history = self._execution_history[-100:]

            return result

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ExecutionResult(
                success=False, output=None, error=f"Execution failed: {e!s}"
            )

    def get_execution_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent execution history."""
        return self._execution_history[-limit:]


# Module-level singleton
_sandbox: ToolSandbox | None = None


def get_sandbox() -> ToolSandbox:
    """Get or create the global tool sandbox."""
    global _sandbox
    if _sandbox is None:
        _sandbox = ToolSandbox()
    return _sandbox


# Convenience functions
async def execute_code(code: str, timeout_seconds: float = 5.0) -> ExecutionResult:
    """Execute Python code in sandbox."""
    sandbox = get_sandbox()
    return await sandbox.execute(
        "python_execute",
        {"code": code},
        limits=ResourceLimits(max_execution_time_seconds=timeout_seconds),
    )


async def calculate(expression: str) -> ExecutionResult:
    """Evaluate a math expression."""
    sandbox = get_sandbox()
    return await sandbox.execute("calculate", {"expression": expression})


async def process_json(
    data: str | dict | list, operation: str = "parse", path: str = ""
) -> ExecutionResult:
    """Process JSON data."""
    sandbox = get_sandbox()
    return await sandbox.execute(
        "json_process", {"data": data, "operation": operation, "path": path}
    )
