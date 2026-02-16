/**
 * Chat Store — Messages, conversations, and streaming state.
 *
 * Owns: conversation list, message history, streaming buffer, active conversation.
 * Does NOT own: auth, theme, model metadata (see modelStore).
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface Citation {
  id: string;
  title: string;
  excerpt?: string;
  score: number;
  url?: string;
}

export interface MessageAttachment {
  name: string;
  url: string;
  type: string;
  size: number;
}

/** Tool invocation event — displayed inline in the message */
export interface ToolEvent {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'success' | 'error';
  input?: Record<string, unknown>;
  output?: string;
  durationMs?: number;
  timestamp: string;
}

/** User feedback on an assistant message */
export interface MessageFeedback {
  rating: 'positive' | 'negative';
  comment?: string;
  timestamp: string;
}

export interface Message {
  id: string;
  conversationId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  attachments?: MessageAttachment[];
  isError?: boolean;
  // Model attribution
  citations?: Citation[];
  modelUsed?: string;
  latencyMs?: number;
  tokenCount?: number;
  /** Which agent/pipeline stage produced this (for multi-agent). */
  agentId?: string;
  /** If this was a fallback model. */
  isFallback?: boolean;
  /** Token-level timing for latency analysis. */
  firstTokenMs?: number;

  // ── Branching ──
  /** Parent message ID — null for root messages */
  parentId?: string;
  /** IDs of alternative branches from the same parent */
  siblingIds?: string[];
  /** Active branch index (for branch navigation) */
  branchIndex?: number;

  // ── Editing ──
  /** Whether this message has been edited */
  isEdited?: boolean;
  /** Original content before editing */
  originalContent?: string;

  // ── Tool usage ──
  /** Tool invocations displayed inline */
  toolEvents?: ToolEvent[];

  // ── Feedback ──
  /** User feedback (thumbs up/down + optional comment) */
  feedback?: MessageFeedback;

  // ── System state ──
  /** Thinking/status text shown while generating */
  thinkingText?: string;
}

export interface Conversation {
  id: string;
  title: string;
  language?: string;
  subject?: string;
  created_at: string;
  updated_at: string;
  /** Summary of the model(s) used in this conversation. */
  modelSummary?: string;
}

// ─── Streaming state ─────────────────────────────────────────────────────────

export type StreamStatus = 'idle' | 'connecting' | 'streaming' | 'cancelling' | 'error';

interface StreamState {
  status: StreamStatus;
  buffer: string;
  /** Accumulated token count during current stream. */
  tokensSoFar: number;
  /** Timestamp when streaming started (for latency calc). */
  startedAt: number | null;
  /** Timestamp of first token arrival. */
  firstTokenAt: number | null;
  /** Error message if status is 'error'. */
  errorMessage: string | null;
}

// ─── Store interface ─────────────────────────────────────────────────────────

interface ChatState {
  // Conversations
  conversations: Conversation[];
  activeConversationId: string | null;
  createConversation: () => Conversation;
  setActiveConversationId: (id: string | null) => void;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;

  // Messages
  messages: Message[];
  addMessage: (message: Message) => void;
  deleteMessage: (id: string) => void;
  deleteMessagesFrom: (id: string) => void;
  replaceLastAssistantMessage: (message: Message) => void;
  /** Efficient update: patch specific fields on a message by id. */
  patchMessage: (id: string, patch: Partial<Message>) => void;

  // ── Branching & Editing ──
  /** Edit a user message, creating a new branch. Returns new branch messages array. */
  editMessage: (messageId: string, newContent: string) => void;
  /** Navigate to a sibling branch */
  switchBranch: (messageId: string, branchIndex: number) => void;

  // ── Feedback ──
  /** Set feedback on a message */
  setFeedback: (messageId: string, feedback: MessageFeedback) => void;

  // ── Tool Events ──
  /** Add or update a tool invocation on a message */
  upsertToolEvent: (messageId: string, event: ToolEvent) => void;

  // Streaming
  stream: StreamState;
  setStreamStatus: (status: StreamStatus) => void;
  appendStreamBuffer: (chunk: string) => void;
  setStreamBuffer: (content: string) => void;
  markFirstToken: () => void;
  startStream: () => void;
  resetStream: () => void;
  setStreamError: (error: string) => void;

  // Legacy compat (delegate to stream.buffer)
  streamingMessage: string;
  setStreamingMessage: (content: string) => void;
}

// ─── Store ───────────────────────────────────────────────────────────────────

const initialStreamState: StreamState = {
  status: 'idle',
  buffer: '',
  tokensSoFar: 0,
  startedAt: null,
  firstTokenAt: null,
  errorMessage: null,
};

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      // ── Conversations ──
      conversations: [],
      activeConversationId: null,

      createConversation: () => {
        const newConv: Conversation = {
          id: `conv-${Date.now()}`,
          title: 'New Chat',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
        set((s) => ({
          conversations: [newConv, ...s.conversations],
          activeConversationId: newConv.id,
          messages: [],
        }));
        return newConv;
      },

      setActiveConversationId: (id) => set({ activeConversationId: id, messages: [] }),

      deleteConversation: (id) => {
        const current = get();
        set({
          conversations: current.conversations.filter((c) => c.id !== id),
          activeConversationId: current.activeConversationId === id ? null : current.activeConversationId,
          messages: current.activeConversationId === id ? [] : current.messages,
        });
      },

      renameConversation: (id, title) => {
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id ? { ...c, title, updated_at: new Date().toISOString() } : c
          ),
        }));
      },

      // ── Messages ──
      messages: [],

      addMessage: (message) => {
        set((s) => ({ messages: [...s.messages, message] }));
      },

      deleteMessage: (id) => {
        set((s) => ({ messages: s.messages.filter((m) => m.id !== id) }));
      },

      deleteMessagesFrom: (id) => {
        set((s) => {
          const idx = s.messages.findIndex((m) => m.id === id);
          return idx === -1 ? s : { messages: s.messages.slice(0, idx) };
        });
      },

      replaceLastAssistantMessage: (message) => {
        set((s) => {
          for (let i = s.messages.length - 1; i >= 0; i--) {
            if (s.messages[i].role === 'assistant') {
              const newMessages = [...s.messages];
              newMessages[i] = message;
              return { messages: newMessages };
            }
          }
          return { messages: [...s.messages, message] };
        });
      },

      patchMessage: (id, patch) => {
        set((s) => ({
          messages: s.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
        }));
      },

      // ── Branching & Editing ──

      editMessage: (messageId, newContent) => {
        set((s) => {
          const msgIndex = s.messages.findIndex((m) => m.id === messageId);
          if (msgIndex === -1) return s;

          const original = s.messages[msgIndex];
          // Save original content on first edit
          const updatedOriginal: Message = {
            ...original,
            originalContent: original.originalContent || original.content,
          };

          // Create the edited version as a new branch
          const editedMessage: Message = {
            ...updatedOriginal,
            id: `${messageId}-edit-${Date.now()}`,
            content: newContent,
            isEdited: true,
            timestamp: new Date().toISOString(),
            parentId: original.parentId || original.id,
            branchIndex: (original.branchIndex ?? 0) + 1,
          };

          // Keep messages up to and including the edited message position,
          // then replace from there with the new edit
          const messagesUpTo = s.messages.slice(0, msgIndex);
          return {
            messages: [...messagesUpTo, editedMessage],
          };
        });
      },

      switchBranch: (messageId, branchIndex) => {
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === messageId ? { ...m, branchIndex } : m
          ),
        }));
      },

      // ── Feedback ──

      setFeedback: (messageId, feedback) => {
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === messageId ? { ...m, feedback } : m
          ),
        }));
      },

      // ── Tool Events ──

      upsertToolEvent: (messageId, event) => {
        set((s) => ({
          messages: s.messages.map((m) => {
            if (m.id !== messageId) return m;
            const existing = m.toolEvents || [];
            const idx = existing.findIndex((e) => e.id === event.id);
            const updated = idx >= 0
              ? existing.map((e, i) => i === idx ? event : e)
              : [...existing, event];
            return { ...m, toolEvents: updated };
          }),
        }));
      },

      // ── Streaming ──
      stream: { ...initialStreamState },

      setStreamStatus: (status) => {
        set((s) => ({ stream: { ...s.stream, status } }));
      },

      appendStreamBuffer: (chunk) => {
        set((s) => ({
          stream: {
            ...s.stream,
            buffer: s.stream.buffer + chunk,
            tokensSoFar: s.stream.tokensSoFar + 1,
          },
          streamingMessage: s.stream.buffer + chunk, // Legacy compat
        }));
      },

      setStreamBuffer: (content) => {
        set((s) => ({
          stream: { ...s.stream, buffer: content },
          streamingMessage: content, // Legacy compat
        }));
      },

      markFirstToken: () => {
        set((s) => {
          if (s.stream.firstTokenAt) return s; // Already marked
          return {
            stream: { ...s.stream, firstTokenAt: performance.now() },
          };
        });
      },

      startStream: () => {
        set({
          stream: {
            status: 'connecting',
            buffer: '',
            tokensSoFar: 0,
            startedAt: performance.now(),
            firstTokenAt: null,
            errorMessage: null,
          },
          streamingMessage: '',
        });
      },

      resetStream: () => {
        set({
          stream: { ...initialStreamState },
          streamingMessage: '',
        });
      },

      setStreamError: (error) => {
        set((s) => ({
          stream: { ...s.stream, status: 'error', errorMessage: error },
        }));
      },

      // ── Legacy compatibility ──
      streamingMessage: '',
      setStreamingMessage: (content) => {
        set((s) => ({
          streamingMessage: content,
          stream: { ...s.stream, buffer: content },
        }));
      },
    }),
    {
      name: 'chat-storage',
      partialize: (state) => ({
        conversations: state.conversations,
        // Don't persist messages or stream state
      }),
    }
  )
);
