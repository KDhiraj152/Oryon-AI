/**
 * Store Barrel — Re-exports all stores with backward-compatible aliases.
 *
 * Architecture (new modular stores):
 *   sessionStore  → auth + profile (persisted)
 *   uiStore       → theme, sidebar, toasts (persisted theme)
 *   chatStore     → conversations, messages, streaming (persisted conversations)
 *   modelStore    → active model, routing, latency (ephemeral)
 *
 * Legacy aliases (useAuthStore, useThemeStore, useProfileStore) delegate
 * to the new stores so existing imports continue to work.
 */

// ── New modular stores ──────────────────────────────────────────────────────

export { useSessionStore } from './sessionStore';
export type { User, UserProfile } from './sessionStore';

export { useUIStore } from './uiStore';
export type { ToastSeverity, ToastItem } from './uiStore';

export { useChatStore } from './chatStore';
export type {
  Citation,
  Message,
  MessageAttachment,
  Conversation,
  StreamStatus,
  ToolEvent,
  MessageFeedback,
} from './chatStore';

export { useModelStore } from './modelStore';
export type {
  ModelInfo,
  RoutingDecision,
  LatencyMetrics,
} from './modelStore';

// ── Legacy backward-compatible aliases ──────────────────────────────────────
// These re-export the new stores under the old names so that every
// existing `import { useAuthStore } from '../store'` keeps working.

export { useSessionStore as useAuthStore } from './sessionStore';
export { useSessionStore as useProfileStore } from './sessionStore';
export { useUIStore as useThemeStore } from './uiStore';
