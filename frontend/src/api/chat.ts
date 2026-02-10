/**
 * Chat API - SSE streaming chat interface
 */

import { API_BASE, getAuthHeader } from './client';
import type { Attachment, ChatContext } from './types';

export interface SendMessageOptions {
  message: string;
  conversationId?: string;
  context?: ChatContext;
  attachments?: { file_id: string; filename: string }[];
  onChunk?: (text: string) => void;
  onStatus?: (status: { stage: string; message: string }) => void;
  onComplete?: (data: { message_id: string; text: string; attachments?: Attachment[] }) => void;
  onError?: (error: string) => void;
}

type StreamCallbacks = Pick<SendMessageOptions, 'onChunk' | 'onStatus' | 'onComplete' | 'onError'>;

/** Process individual SSE stream lines */
function processStreamLine(line: string, callbacks: StreamCallbacks): void {
  if (!line.startsWith('data: ')) return;

  try {
    const data = JSON.parse(line.slice(6));
    switch (data.type) {
      case 'chunk':
        callbacks.onChunk?.(data.data.text);
        break;
      case 'status':
        callbacks.onStatus?.(data.data);
        break;
      case 'complete':
        callbacks.onComplete?.(data.data);
        break;
      case 'error':
        callbacks.onError?.(data.data.error);
        break;
    }
  } catch {
    // Invalid JSON, skip
  }
}

/** Read and process SSE stream */
async function readStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  callbacks: StreamCallbacks
): Promise<void> {
  const decoder = new TextDecoder();
  let buffer = '';
  let done = false;

  while (!done) {
    const result = await reader.read();
    done = result.done;
    if (done) break;

    buffer += decoder.decode(result.value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      processStreamLine(line, callbacks);
    }
  }
}

export const chat = {
  /**
   * Send a message with SSE streaming response
   * @returns Abort function to cancel the request
   */
  sendMessage(options: SendMessageOptions): () => void {
    const { message, conversationId, context, attachments, onChunk, onStatus, onComplete, onError } = options;
    const abortController = new AbortController();

    const body = JSON.stringify({
      message,
      conversation_id: conversationId,
      context,
      attachments,
    });

    fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...getAuthHeader(),
      },
      body,
      signal: abortController.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const error = await response.json().catch(() => ({ detail: 'Chat request failed' }));
          onError?.(error.detail || 'Chat request failed');
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          onError?.('No response body');
          return;
        }

        await readStream(reader, { onChunk, onStatus, onComplete, onError });
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          onError?.(error.message);
        }
      });

    return () => abortController.abort();
  },
};
