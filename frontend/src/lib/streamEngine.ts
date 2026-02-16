/**
 * Stream Engine — Single, canonical SSE/JSON stream reader.
 *
 * Replaces the three duplicate implementations in:
 *   - lib/chatUtils.ts   (parseSSEStream / parseJSONResponse)
 *   - api/client.ts      (processStreamLine / readStream)
 *   - api/chat.ts        (inline SSE parsing)
 *
 * Features:
 *   - Token-by-token callback (onToken) for granular rendering
 *   - Interrupt / cancel via AbortSignal
 *   - First-token timing for TTFT metrics
 *   - Meta extraction (model, latency, citations)
 *   - Automatic fallback from SSE → plain JSON
 */

import type { Citation } from '../store/chatStore';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface StreamMeta {
  model?: string;
  latencyMs?: number;
  citations?: Citation[];
  tokenCount?: number;
  agentId?: string;
  isFallback?: boolean;
}

export interface StreamCallbacks {
  /** Called for each token/chunk — use for token-by-token rendering. */
  onToken: (accumulated: string, chunk: string) => void;
  /** Called once when the full response is assembled. */
  onComplete?: (text: string, meta: StreamMeta) => void;
  /** Called on non-fatal SSE error events from the server. */
  onServerError?: (error: string) => void;
  /** Called when the first token arrives (for TTFT tracking). */
  onFirstToken?: () => void;
  /** Called on status events (multi-agent pipeline stages). */
  onStatus?: (stage: string, message: string) => void;
}

export interface StreamResult {
  text: string;
  meta: StreamMeta;
  /** Was the stream cancelled by the user? */
  cancelled: boolean;
}

// ─── SSE Stream Reader ──────────────────────────────────────────────────────

/**
 * Read a `text/event-stream` response body token-by-token.
 *
 * Protocol:
 *   data: {"type":"chunk","data":{"text":"..."}}
 *   data: {"type":"status","data":{"stage":"...","message":"..."}}
 *   data: {"type":"complete","data":{"text":"...","model":"...","latency_ms":...}}
 *   data: {"type":"error","data":{"error":"..."}}
 *   data: [DONE]
 */
export async function readSSEStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<StreamResult> {
  const decoder = new TextDecoder();
  let buffer = '';
  let accumulated = '';
  let meta: StreamMeta = {};
  let firstTokenFired = false;
  let cancelled = false;

  // Abort handling
  if (signal) {
    const onAbort = () => {
      cancelled = true;
      reader.cancel().catch(() => {});
    };
    if (signal.aborted) {
      return { text: accumulated, meta, cancelled: true };
    }
    signal.addEventListener('abort', onAbort, { once: true });
  }

  try {
    while (true) {
      if (cancelled) break;

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (cancelled) break;
        if (!line.startsWith('data: ')) continue;

        const payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;

        try {
          const data = JSON.parse(payload);

          switch (data.type) {
            case 'chunk': {
              const chunk = data.data?.text ?? '';
              if (chunk) {
                if (!firstTokenFired) {
                  firstTokenFired = true;
                  callbacks.onFirstToken?.();
                }
                accumulated += chunk;
                callbacks.onToken(accumulated, chunk);
              }
              break;
            }

            case 'complete': {
              if (data.data?.text) {
                accumulated = data.data.text;
                callbacks.onToken(accumulated, '');
              }
              meta = {
                model: data.data?.model,
                latencyMs: data.data?.latency_ms,
                citations: data.data?.citations,
                tokenCount: data.data?.token_count,
                agentId: data.data?.agent_id,
                isFallback: data.data?.is_fallback,
              };
              break;
            }

            case 'status': {
              callbacks.onStatus?.(data.data?.stage ?? '', data.data?.message ?? '');
              break;
            }

            case 'error': {
              callbacks.onServerError?.(data.data?.error ?? 'Unknown stream error');
              break;
            }
          }
        } catch {
          // Non-JSON line — skip
        }
      }
    }

    // Process remaining buffer
    if (buffer.startsWith('data: ') && !cancelled) {
      const payload = buffer.slice(6).trim();
      if (payload !== '[DONE]') {
        try {
          const data = JSON.parse(payload);
          if (data.type === 'chunk' && data.data?.text) {
            accumulated += data.data.text;
            callbacks.onToken(accumulated, data.data.text);
          }
        } catch {
          // Ignore
        }
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // Already released
    }
  }

  callbacks.onComplete?.(accumulated, meta);
  return { text: accumulated, meta, cancelled };
}

// ─── JSON (non-SSE) Stream Reader ───────────────────────────────────────────

/**
 * Read a plain JSON response body (used for guest/unauthenticated endpoints).
 */
export async function readJSONStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<StreamResult> {
  const decoder = new TextDecoder();
  let raw = '';
  let cancelled = false;

  if (signal) {
    const onAbort = () => {
      cancelled = true;
      reader.cancel().catch(() => {});
    };
    if (signal.aborted) {
      return { text: '', meta: {}, cancelled: true };
    }
    signal.addEventListener('abort', onAbort, { once: true });
  }

  try {
    while (true) {
      if (cancelled) break;
      const { done, value } = await reader.read();
      if (done) break;
      raw += decoder.decode(value, { stream: true });
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // Already released
    }
  }

  if (cancelled) {
    return { text: raw, meta: {}, cancelled: true };
  }

  let text = raw;
  let meta: StreamMeta = {};

  try {
    const data = JSON.parse(raw);
    text = data.response || data.text || data.message || raw;
    meta = {
      model: data.model,
      latencyMs: data.latency_ms,
      citations: data.citations,
      tokenCount: data.token_count,
    };
  } catch {
    // Not JSON — treat as plain text
  }

  callbacks.onFirstToken?.();
  callbacks.onToken(text, text);
  callbacks.onComplete?.(text, meta);

  return { text, meta, cancelled: false };
}

// ─── Unified entry point ────────────────────────────────────────────────────

/**
 * Read a response stream, auto-detecting SSE vs JSON based on content type.
 */
export async function readResponseStream(
  response: Response,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<StreamResult> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Response body is not readable');
  }

  const contentType = response.headers.get('content-type') || '';
  const isSSE = contentType.includes('text/event-stream');

  return isSSE
    ? readSSEStream(reader, callbacks, signal)
    : readJSONStream(reader, callbacks, signal);
}
