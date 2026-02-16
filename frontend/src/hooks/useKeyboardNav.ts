/**
 * useKeyboardNav — Keyboard navigation for the chat interface.
 *
 * Shortcuts:
 *   Escape        → Cancel streaming / close sidebar
 *   Cmd/Ctrl+Shift+C → Copy last assistant message
 *   Cmd/Ctrl+K    → Focus chat input
 *   Cmd/Ctrl+/    → Toggle sidebar
 *   Cmd/Ctrl+N    → New conversation
 *   Cmd/Ctrl+Shift+Backspace → Delete conversation
 *   ArrowUp       → Edit last user message (when input is empty and focused)
 *
 * All shortcuts respect active element context (won't fire inside textareas
 * unless explicitly intended).
 */

import { useEffect, useCallback, useRef } from 'react';

interface KeyboardNavOptions {
  /** Cancel current stream */
  onCancel?: () => void;
  /** Focus the chat input */
  onFocusInput?: () => void;
  /** Toggle sidebar */
  onToggleSidebar?: () => void;
  /** Create new conversation */
  onNewChat?: () => void;
  /** Copy last assistant message */
  onCopyLast?: () => void;
  /** Edit last user message */
  onEditLast?: () => void;
  /** Whether streaming is active (Escape → cancel) */
  isStreaming?: boolean;
  /** Whether sidebar is open (Escape → close) */
  isSidebarOpen?: boolean;
  /** Close sidebar */
  onCloseSidebar?: () => void;
  /** Whether the hook is enabled */
  enabled?: boolean;
}

export function useKeyboardNav({
  onCancel,
  onFocusInput,
  onToggleSidebar,
  onNewChat,
  onCopyLast,
  onEditLast,
  isStreaming = false,
  isSidebarOpen = false,
  onCloseSidebar,
  enabled = true,
}: KeyboardNavOptions): void {
  // Use refs for callbacks to avoid re-registering the listener
  const callbacks = useRef({
    onCancel, onFocusInput, onToggleSidebar, onNewChat,
    onCopyLast, onEditLast, onCloseSidebar,
  });
  callbacks.current = {
    onCancel, onFocusInput, onToggleSidebar, onNewChat,
    onCopyLast, onEditLast, onCloseSidebar,
  };

  const stateRef = useRef({ isStreaming, isSidebarOpen });
  stateRef.current = { isStreaming, isSidebarOpen };

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    const meta = e.metaKey || e.ctrlKey;
    const target = e.target as HTMLElement;
    const isInputFocused = target.tagName === 'TEXTAREA' || target.tagName === 'INPUT' ||
                           target.isContentEditable;

    // Escape — always available
    if (e.key === 'Escape') {
      e.preventDefault();
      if (stateRef.current.isStreaming) {
        callbacks.current.onCancel?.();
      } else if (stateRef.current.isSidebarOpen) {
        callbacks.current.onCloseSidebar?.();
      }
      return;
    }

    // Cmd/Ctrl shortcuts — don't intercept when user is typing normally
    if (meta) {
      // Cmd+K → Focus input
      if (e.key === 'k') {
        e.preventDefault();
        callbacks.current.onFocusInput?.();
        return;
      }

      // Cmd+/ → Toggle sidebar
      if (e.key === '/') {
        e.preventDefault();
        callbacks.current.onToggleSidebar?.();
        return;
      }

      // Cmd+N → New chat (only when not in input)
      if (e.key === 'n' && !isInputFocused) {
        e.preventDefault();
        callbacks.current.onNewChat?.();
        return;
      }

      // Cmd+Shift+C → Copy last response
      if (e.key === 'C' && e.shiftKey) {
        e.preventDefault();
        callbacks.current.onCopyLast?.();
        return;
      }
    }

    // ArrowUp in empty input → edit last message
    if (e.key === 'ArrowUp' && isInputFocused && target.tagName === 'TEXTAREA') {
      const textarea = target as HTMLTextAreaElement;
      if (textarea.value === '' && textarea.selectionStart === 0) {
        e.preventDefault();
        callbacks.current.onEditLast?.();
      }
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [enabled, handleKeyDown]);
}
