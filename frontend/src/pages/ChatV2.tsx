/**
 * Chat — Root chat page.
 *
 * Integrations in this rewrite:
 * - useSmartScroll (RAF-driven 60fps auto-scroll with user intent detection)
 * - useKeyboardNav (Escape, Cmd+K, Cmd+/, Cmd+N, Cmd+Shift+C, ArrowUp)
 * - ChatMessageV2 (all new sub-components: FeedbackButtons, ToolInvocation,
 *   InlineEditor, BranchNavigator, CollapsibleContent, SourcePreview, StreamingMarkdown)
 * - MessageSkeleton (skeleton loader for connecting state)
 * - SystemFeedback (adaptive status with elapsed timer)
 * - LatencyIndicator + CancelButton (streaming toolbar)
 * - ErrorPanel (centralized error display)
 * - Staggered message entrance animations
 */

import { useState, useEffect, useCallback, memo, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import { useSessionStore } from '../store/sessionStore';
import { useUIStore } from '../store/uiStore';
import { ChatMessage } from '../components/chat/ChatMessageV2';
import ChatInput from '../components/chat/ChatInput';
import { Toast, type ToastType } from '../components/ui/Toast';
import { EmptyState } from '../components/chat/EmptyState';
import { MessageSkeleton } from '../components/chat/MessageSkeleton';
import { SystemFeedback } from '../components/chat/SystemFeedback';
import { useAudioPlayback } from '../hooks/useChat';
import { useStreamingChat } from '../hooks/useStreamingChat';
import { useSmartScroll } from '../hooks/useSmartScroll';
import { useKeyboardNav } from '../hooks/useKeyboardNav';
import { CancelButton } from '../components/chat/CancelButton';
import { LatencyIndicator } from '../components/chat/LatencyIndicator';
import { ArrowDown } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';

// ─── Scroll-to-bottom Button (memoized) ──────────────────────────────────────

const ScrollToBottomButton = memo(function ScrollToBottomButton({
  onClick,
  isDark,
}: {
  onClick: () => void;
  isDark: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={`absolute bottom-28 left-1/2 -translate-x-1/2 p-2.5 rounded-full shadow-lg transition-all duration-200 z-10
        animate-fade-in
        ${isDark
          ? 'bg-white/[0.08] hover:bg-white/[0.12] text-white/60 border border-white/[0.06]'
          : 'bg-white hover:bg-gray-50 text-gray-400 border border-gray-200/80 shadow-sm'}`}
      title="Scroll to bottom"
      aria-label="Scroll to bottom"
    >
      <ArrowDown className="w-4 h-4" />
    </button>
  );
});

// ─── Chat Page ───────────────────────────────────────────────────────────────

export default function Chat() {
  // ── Local UI state ──
  const [selectedLanguage, setSelectedLanguage] = useState('auto');
  const [toast, setToast] = useState<{ message: string; type: ToastType } | null>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [regeneratingMessageId, setRegeneratingMessageId] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // ── Store selectors (decoupled state layers) ──
  const { activeConversationId, messages, streamingMessage } = useChatStore(
    useShallow((s) => ({
      activeConversationId: s.activeConversationId,
      messages: s.messages,
      streamingMessage: s.streamingMessage,
    })),
  );

  const streamStatus = useChatStore((s) => s.stream.status);
  const editMessage = useChatStore((s) => s.editMessage);
  const switchBranch = useChatStore((s) => s.switchBranch);
  const setActiveConversationId = useChatStore((s) => s.setActiveConversationId);

  const isAuthenticated = useSessionStore((s) => s.isAuthenticated);
  const fetchProfile = useSessionStore((s) => s.fetchProfile);

  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const setSidebarOpen = useUIStore((s) => s.setSidebarOpen);

  // ── Streaming hook ──
  const { send, cancel, retry, isStreaming, isConnecting } = useStreamingChat(selectedLanguage);

  // ── Toast helpers ──
  const showToast = useCallback((message: string, type: ToastType = 'error') => {
    setToast({ message, type });
  }, []);
  const hideToast = useCallback(() => setToast(null), []);

  // ── Audio ──
  const { playingAudioMessageId, handleAudio, stopAudio } = useAudioPlayback({
    selectedLanguage,
    showToast,
  });

  // ── Smart scroll (replaces useChatScroll) ──
  const {
    containerRef,
    bottomRef,
    showScrollButton,
    scrollToBottom,
  } = useSmartScroll({
    messageCount: messages.length,
    streamingContent: streamingMessage || '',
    isStreaming: isStreaming || isConnecting,
  });

  // ── Keyboard navigation ──
  const handleFocusInput = useCallback(() => {
    // Try ref first, then fallback to DOM query
    if (inputRef.current) {
      inputRef.current.focus();
    } else {
      const textarea = document.querySelector<HTMLTextAreaElement>(
        'textarea[placeholder]',
      );
      textarea?.focus();
    }
  }, []);

  const handleCopyLast = useCallback(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
    if (lastAssistant) {
      navigator.clipboard.writeText(lastAssistant.content);
      showToast('Copied to clipboard', 'success');
    }
  }, [messages, showToast]);

  const handleNewChat = useCallback(() => {
    setActiveConversationId(null);
    handleFocusInput();
  }, [setActiveConversationId, handleFocusInput]);

  const handleEditLast = useCallback(() => {
    // Find last user message — the InlineEditor will activate on it
    // For now, just scroll up to it (InlineEditor is triggered via ChatMessage)
  }, []);

  useKeyboardNav({
    onCancel: cancel,
    onFocusInput: handleFocusInput,
    onToggleSidebar: toggleSidebar,
    onNewChat: handleNewChat,
    onCopyLast: handleCopyLast,
    onEditLast: handleEditLast,
    isStreaming: isStreaming || isConnecting,
    isSidebarOpen: sidebarOpen,
    onCloseSidebar: () => setSidebarOpen(false),
    enabled: true,
  });

  // ── Load profile on mount ──
  useEffect(() => {
    if (isAuthenticated) fetchProfile();
  }, [isAuthenticated, fetchProfile]);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      cancel();
      stopAudio();
    };
  }, [cancel, stopAudio]);

  // ── Handlers ──
  const handleSend = useCallback(
    async (message: string, files?: File[], language?: string) => {
      await send({ message, files, language });
      // Ensure we scroll to bottom on own message
      scrollToBottom(true);
    },
    [send, scrollToBottom],
  );

  const handleCopy = useCallback((content: string) => {
    navigator.clipboard.writeText(content);
  }, []);

  const handleRetry = useCallback(
    async (messageId?: string) => {
      setIsRegenerating(true);
      setRegeneratingMessageId(messageId || null);
      try {
        await retry(messageId);
      } finally {
        setIsRegenerating(false);
        setRegeneratingMessageId(null);
      }
    },
    [retry],
  );

  const handleEdit = useCallback(
    (messageId: string, newContent: string) => {
      editMessage(messageId, newContent);
      // After editing, re-send the conversation from that point
      // The editMessage action slices messages, so the last message is the edited one
      send({ message: newContent, language: selectedLanguage });
    },
    [editMessage, send, selectedLanguage],
  );

  const handleBranchNavigate = useCallback(
    (messageId: string, branchIndex: number) => {
      switchBranch(messageId, branchIndex);
    },
    [switchBranch],
  );

  const handleError = useCallback(
    (error: string) => showToast(error, 'error'),
    [showToast],
  );

  const handleQuickAction = useCallback(
    (prompt: string) => send({ message: prompt, language: selectedLanguage }),
    [send, selectedLanguage],
  );

  // ── Derived state ──
  const isThinking = isConnecting && !streamingMessage;
  const showEmptyState = messages.length === 0 && !streamingMessage && !isThinking;

  return (
    <div
      className={`flex h-full overflow-hidden ${isDark ? 'bg-[#0a0a0a] text-white' : 'bg-[#fafafa] text-gray-900'}`}
    >
      {/* Toast */}
      {toast && <Toast message={toast.message} type={toast.type} onClose={hideToast} />}

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 w-full relative overflow-hidden">
        {/* Messages area */}
        {showEmptyState ? (
          <div className="flex-1 overflow-y-auto overflow-x-hidden">
            <EmptyState isDark={isDark} onQuickAction={handleQuickAction} />
          </div>
        ) : (
          <div
            ref={containerRef as React.RefObject<HTMLDivElement>}
            className="flex-1 overflow-y-auto overflow-x-hidden scroll-smooth"
          >
            <div
              className="w-full max-w-3xl mx-auto px-4 py-6"
              id="main-content"
              role="log"
              aria-live="polite"
              aria-label="Chat messages"
            >
              {/* Message list with stagger animations */}
              {messages.map((msg, index) => (
                <ChatMessage
                  key={msg.id}
                  message={msg}
                  isStreaming={isRegenerating && regeneratingMessageId === msg.id}
                  isPlayingAudio={playingAudioMessageId === msg.id}
                  staggerIndex={index < 20 ? index : undefined}
                  onCopy={() => handleCopy(msg.content)}
                  onRetry={msg.role === 'assistant' ? (id) => handleRetry(id) : undefined}
                  onAudio={() => handleAudio(msg.content, msg.id)}
                  onEdit={msg.role === 'user' ? handleEdit : undefined}
                  onBranchNavigate={handleBranchNavigate}
                />
              ))}

              {/* ── Footer: Skeleton / Streaming / System feedback ── */}
              
              {/* Thinking skeleton — before first token */}
              {isThinking && !isRegenerating && <MessageSkeleton />}

              {/* Active streaming message — token-by-token */}
              {streamingMessage && !isRegenerating && (
                <div aria-live="polite" aria-atomic="false">
                  <ChatMessage
                    message={{
                      id: 'streaming',
                      conversationId: activeConversationId || '',
                      role: 'assistant',
                      content: streamingMessage,
                      timestamp: new Date().toISOString(),
                    }}
                    isStreaming
                  />
                </div>
              )}

              {/* Regenerating state */}
              {isRegenerating && streamingMessage && (
                <div aria-live="polite" aria-atomic="false">
                  <ChatMessage
                    message={{
                      id: 'regenerating',
                      conversationId: activeConversationId || '',
                      role: 'assistant',
                      content: streamingMessage,
                      timestamp: new Date().toISOString(),
                    }}
                    isStreaming
                  />
                </div>
              )}

              {/* System feedback — adaptive status during streaming */}
              {(isStreaming || isConnecting) && (
                <SystemFeedback />
              )}

              {/* Bottom sentinel for scroll tracking */}
              <div ref={bottomRef as React.RefObject<HTMLDivElement>} className="h-6" />
            </div>
          </div>
        )}

        {/* Scroll-to-bottom button */}
        {showScrollButton && <ScrollToBottomButton onClick={() => scrollToBottom()} isDark={isDark} />}

        {/* Input area — fixed at bottom */}
        <div className={`flex-shrink-0 py-4 px-4 ${isDark ? 'bg-[#0a0a0a]' : 'bg-[#fafafa]'}`}>
          <div className="w-full max-w-3xl mx-auto">
            {/* Latency + Cancel bar during streaming */}
            {(isStreaming || isConnecting) && (
              <div className="flex items-center justify-between mb-3 px-1 animate-fade-in">
                <LatencyIndicator />
                <CancelButton
                  isStreaming={isStreaming}
                  isCancelling={streamStatus === 'cancelling'}
                  onCancel={cancel}
                />
              </div>
            )}

            <ChatInput
              onSend={handleSend}
              selectedLanguage={selectedLanguage}
              onLanguageChange={setSelectedLanguage}
              disabled={isStreaming || isConnecting}
              onError={handleError}
            />

            {/* Footer text */}
            <p
              className={`text-center text-[11px] mt-3 font-medium tracking-wide
                ${isDark ? 'text-white/20' : 'text-black/25'}`}
            >
              Oryon can make mistakes. Verify important information.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
