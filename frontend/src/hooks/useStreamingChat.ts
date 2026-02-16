/**
 * useStreamingChat — Unified hook for sending messages with streaming,
 * cancellation, latency tracking, and model attribution.
 *
 * Replaces the inline fetch logic in Chat.tsx with a clean, reusable hook.
 */

import { useCallback, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import { useSessionStore } from '../store/sessionStore';
import { useModelStore } from '../store/modelStore';
import { useUIStore } from '../store/uiStore';
import { readResponseStream, type StreamMeta } from '../lib/streamEngine';
import {
  processUploadedFiles,
  createUserMessage,
  createAssistantMessage,
  createErrorMessage,
  buildChatHeaders,
  buildChatRequestBody,
  buildConversationHistory,
  getChatEndpoint,
  getResponseLanguage,
} from '../lib/chatUtils';

// ─── Types ───────────────────────────────────────────────────────────────────

interface SendOptions {
  message: string;
  files?: File[];
  language?: string;
}

interface UseStreamingChatReturn {
  /** Send a new message (with optional files). */
  send: (opts: SendOptions) => Promise<void>;
  /** Cancel the current stream. */
  cancel: () => void;
  /** Retry / regenerate the last assistant response. */
  retry: (messageId?: string) => Promise<void>;
  /** Whether a stream is currently active. */
  isStreaming: boolean;
  /** Whether we're waiting for the first token. */
  isConnecting: boolean;
}

// ─── Hook ────────────────────────────────────────────────────────────────────

export function useStreamingChat(selectedLanguage: string): UseStreamingChatReturn {
  const abortRef = useRef<AbortController | null>(null);

  // Store selectors (stable references)
  const addMessage = useChatStore((s) => s.addMessage);
  const replaceLastAssistantMessage = useChatStore((s) => s.replaceLastAssistantMessage);
  const createConversation = useChatStore((s) => s.createConversation);
  const startStream = useChatStore((s) => s.startStream);
  const resetStream = useChatStore((s) => s.resetStream);
  const setStreamStatus = useChatStore((s) => s.setStreamStatus);
  const setStreamBuffer = useChatStore((s) => s.setStreamBuffer);
  const markFirstToken = useChatStore((s) => s.markFirstToken);
  const setStreamError = useChatStore((s) => s.setStreamError);
  const streamStatus = useChatStore((s) => s.stream.status);

  const accessToken = useSessionStore((s) => s.accessToken);
  const isAuthenticated = useSessionStore((s) => s.isAuthenticated);

  const recordLatency = useModelStore((s) => s.recordLatency);
  const setActiveModel = useModelStore((s) => s.setActiveModel);
  const addRoutingDecision = useModelStore((s) => s.addRoutingDecision);
  const setFallback = useModelStore((s) => s.setFallback);

  const addToast = useUIStore((s) => s.addToast);

  // ── Cancel ──
  const cancel = useCallback(() => {
    if (abortRef.current && !abortRef.current.signal.aborted) {
      setStreamStatus('cancelling');
      abortRef.current.abort();
    }
  }, [setStreamStatus]);

  // ── Core send logic ──
  const executeSend = useCallback(
    async (
      fullMessage: string,
      convId: string,
      history: { role: string; content: string }[],
      responseLanguage: string | undefined,
    ): Promise<{ text: string; meta: StreamMeta; cancelled: boolean }> => {
      abortRef.current = new AbortController();
      const signal = abortRef.current.signal;
      const sendStartedAt = performance.now();

      startStream();

      const response = await fetch(getChatEndpoint(isAuthenticated), {
        method: 'POST',
        headers: buildChatHeaders(accessToken ?? undefined),
        body: buildChatRequestBody({
          message: fullMessage,
          conversationId: convId,
          history,
          language: responseLanguage,
          isAuthenticated,
        }),
        signal,
      });

      // Handle 401 fallback for authenticated users
      if (response.status === 401 && isAuthenticated) {
        const guestResponse = await fetch('/api/v2/chat/guest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: fullMessage, language: responseLanguage }),
          signal,
        });
        if (!guestResponse.ok) {
          throw new Error(`HTTP error! status: ${guestResponse.status}`);
        }
        setFallback(true, 'Auth token expired — using guest endpoint');
        addToast('Session expired. Response served via guest mode.', 'warning', 4000);

        return readResponseStream(guestResponse, {
          onToken: (accumulated) => {
            setStreamStatus('streaming');
            setStreamBuffer(accumulated);
          },
          onFirstToken: () => markFirstToken(),
        }, signal);
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await readResponseStream(response, {
        onToken: (accumulated) => {
          setStreamStatus('streaming');
          setStreamBuffer(accumulated);
        },
        onFirstToken: () => {
          markFirstToken();
          setStreamStatus('streaming');
        },
        onServerError: (error) => {
          addToast(error, 'error');
        },
        onStatus: (stage, message) => {
          // Multi-agent status updates — update toast
          addToast(`${stage}: ${message}`, 'info', 3000);
        },
      }, signal);

      // Record latency metrics
      const totalMs = performance.now() - sendStartedAt;
      const streamState = useChatStore.getState().stream;
      const ttft = streamState.firstTokenAt && streamState.startedAt
        ? streamState.firstTokenAt - streamState.startedAt
        : null;
      const tokensPerSecond = streamState.tokensSoFar > 0 && totalMs > 0
        ? (streamState.tokensSoFar / totalMs) * 1000
        : null;

      recordLatency(ttft, totalMs, tokensPerSecond);

      // Record routing decision
      if (result.meta.model) {
        setActiveModel({ id: result.meta.model, name: result.meta.model });
        addRoutingDecision({
          timestamp: Date.now(),
          requestedModel: 'auto',
          servedBy: result.meta.model,
          reason: result.meta.isFallback ? 'fallback-load' : 'primary',
          latencyMs: result.meta.latencyMs ?? totalMs,
        });
        if (result.meta.isFallback) {
          setFallback(true, 'Backend routed to fallback model');
        } else {
          setFallback(false);
        }
      }

      return result;
    },
    [
      isAuthenticated, accessToken, startStream, setStreamStatus,
      setStreamBuffer, markFirstToken, recordLatency,
      setActiveModel, addRoutingDecision, setFallback, addToast,
    ]
  );

  // ── Public: send ──
  const send = useCallback(
    async ({ message, files, language }: SendOptions) => {
      if (!message.trim() && !files?.length) return;

      const responseLanguage = getResponseLanguage(language, selectedLanguage);
      let convId = useChatStore.getState().activeConversationId;
      if (!convId) {
        convId = createConversation().id;
      }

      // Process files
      let fileContext = '';
      if (files?.length) {
        const result = await processUploadedFiles(files);
        for (const err of result.errors) {
          addToast(err, 'error');
        }
        fileContext = result.context;
      }

      const fullMessage = fileContext ? `${message}\n${fileContext}` : message;
      const messages = useChatStore.getState().messages;
      const history = buildConversationHistory(messages);

      // Optimistic: add user message immediately
      addMessage(createUserMessage(convId, message, files));

      try {
        const { text, meta, cancelled } = await executeSend(
          fullMessage, convId, history, responseLanguage,
        );

        resetStream();

        if (cancelled) return;

        // Stream state already captured by modelStore latency tracking

        addMessage(createAssistantMessage(
          convId,
          text || 'I apologize, but I was unable to generate a response. Please try again.',
          !text,
          {
            citations: meta.citations,
            modelUsed: meta.model,
            latencyMs: meta.latencyMs,
            tokenCount: meta.tokenCount,
          },
        ));
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          resetStream();
          return;
        }
        console.error('Chat error:', error);
        setStreamError(error instanceof Error ? error.message : 'Unknown error');
        resetStream();
        addMessage(createErrorMessage(convId));
      }
    },
    [selectedLanguage, executeSend, addMessage, createConversation, resetStream, setStreamError, addToast],
  );

  // ── Public: retry ──
  const retry = useCallback(
    async (messageId?: string) => {
      const currentMessages = useChatStore.getState().messages;
      let userMessageToRetry: (typeof currentMessages)[0] | undefined;

      if (messageId) {
        const idx = currentMessages.findIndex((m) => m.id === messageId);
        if (idx > 0 && currentMessages[idx - 1]?.role === 'user') {
          userMessageToRetry = currentMessages[idx - 1];
        }
      }

      if (!userMessageToRetry) {
        const userMessages = currentMessages.filter((m) => m.role === 'user');
        userMessageToRetry = userMessages[userMessages.length - 1];
      }

      if (!userMessageToRetry) return;

      const convId = useChatStore.getState().activeConversationId ?? createConversation().id;
      const history = currentMessages
        .filter((m) => m.id !== messageId)
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }));

      const langToSend = selectedLanguage === 'auto' ? undefined : selectedLanguage;

      try {
        const { text, meta, cancelled } = await executeSend(
          userMessageToRetry.content, convId, history, langToSend,
        );

        resetStream();
        if (cancelled) return;

        const newMsg = createAssistantMessage(
          convId,
          text || 'I apologize, but I was unable to generate a response. Please try again.',
          !text,
          { modelUsed: meta.model, latencyMs: meta.latencyMs, tokenCount: meta.tokenCount },
        );

        if (messageId) {
          // Replace the message being retried
          newMsg.id = messageId;
          replaceLastAssistantMessage(newMsg);
        } else {
          addMessage(newMsg);
        }
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          resetStream();
          return;
        }
        console.error('Retry error:', error);
        resetStream();
        addToast('Failed to regenerate response. Please try again.', 'error');
      }
    },
    [selectedLanguage, executeSend, createConversation, resetStream, replaceLastAssistantMessage, addMessage, addToast],
  );

  return {
    send,
    cancel,
    retry,
    isStreaming: streamStatus === 'streaming' || streamStatus === 'cancelling',
    isConnecting: streamStatus === 'connecting',
  };
}
