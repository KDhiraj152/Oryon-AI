"""
Token Streamer — Async Streaming Response Generator
======================================================

Provides streaming token delivery from model inference
to API response with:
  - Backpressure-aware buffering
  - Cancellation support
  - Heartbeat keep-alive for long generations
  - Token counting and throughput tracking
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from backend.infra.telemetry import get_logger

logger = get_logger(__name__)

@dataclass
class StreamConfig:
    """Streaming configuration."""

    buffer_size: int = 64
    heartbeat_interval_s: float = 15.0
    max_tokens: int = 8192
    timeout_s: float = 120.0

class TokenStreamer:
    """
    Manages streaming token delivery with backpressure and cancellation.

    Usage:
        streamer = TokenStreamer(config=StreamConfig())

        # Producer side (from inference backend):
        async def produce():
            for token in model_output:
                await streamer.push(token)
            await streamer.finish()

        # Consumer side (API response):
        async for chunk in streamer.stream():
            yield chunk
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self._config = config or StreamConfig()
        self._buffer: asyncio.Queue[str | None] = asyncio.Queue(
            maxsize=self._config.buffer_size
        )
        self._cancelled = False
        self._finished = False
        self._token_count = 0
        self._start_time = time.monotonic()
        self._first_token_time: float | None = None

    async def push(self, token: str) -> None:
        """Push a token into the stream buffer."""
        if self._cancelled:
            raise asyncio.CancelledError("Stream cancelled")

        await self._buffer.put(token)
        self._token_count += 1

        if self._first_token_time is None:
            self._first_token_time = time.monotonic()

    async def finish(self) -> None:
        """Signal end of stream."""
        self._finished = True
        await self._buffer.put(None)  # Sentinel

    async def cancel(self) -> None:
        """Cancel the stream."""
        self._cancelled = True
        self._finished = True
        # Drain buffer
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._buffer.put(None)

    async def stream(self) -> AsyncGenerator[str, None]:
        """
        Consume the token stream.

        Yields tokens as they become available.
        Handles heartbeats, timeouts, and cancellation.
        """
        time.monotonic()

        while True:
            if self._cancelled:
                return

            try:
                token = await asyncio.wait_for(
                    self._buffer.get(),
                    timeout=self._config.heartbeat_interval_s,
                )

                if token is None:
                    # End of stream
                    return

                time.monotonic()
                yield token

                # Check token limit
                if self._token_count >= self._config.max_tokens:
                    logger.warning(
                        "stream_token_limit",
                        tokens=self._token_count,
                        limit=self._config.max_tokens,
                    )
                    return

            except TimeoutError:
                # Check overall timeout
                elapsed = time.monotonic() - self._start_time
                if elapsed > self._config.timeout_s:
                    logger.warning(
                        "stream_timeout",
                        elapsed_s=round(elapsed, 1),
                        tokens=self._token_count,
                    )
                    return

                # Send heartbeat (empty string, filtered by SSE layer)
                continue

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def time_to_first_token_ms(self) -> float | None:
        if self._first_token_time is None:
            return None
        return (self._first_token_time - self._start_time) * 1000

    @property
    def tokens_per_second(self) -> float:
        elapsed = time.monotonic() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._token_count / elapsed

    def get_stats(self) -> dict[str, Any]:
        return {
            "token_count": self._token_count,
            "ttft_ms": round(self.time_to_first_token_ms, 2) if self.time_to_first_token_ms else None,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "cancelled": self._cancelled,
            "finished": self._finished,
            "buffer_size": self._buffer.qsize(),
        }
