import { useState, useEffect, useRef, useCallback, memo } from 'react';
import { useChatStore, useThemeStore, useAuthStore, useProfileStore, shallow } from '../store';
import type { Message, Citation } from '../store';
import ChatMessage from '../components/chat/ChatMessage';
import ChatInput from '../components/chat/ChatInput';
import { Toast, type ToastType } from '../components/chat/Toast';
import { EmptyState } from '../components/chat/EmptyState';
import { ThinkingIndicator, RegeneratingIndicator } from '../components/chat/ChatIndicators';
import { useAudioPlayback, useChatScroll } from '../hooks/useChat';
import { ArrowDown } from 'lucide-react';
import 'katex/dist/katex.min.css';
import {
  processUploadedFiles,
  parseSSEStream,
  parseJSONResponse,
  createUserMessage,
  createAssistantMessage,
  createErrorMessage,
  buildChatHeaders,
  buildChatRequestBody,
  buildConversationHistory,
  getChatEndpoint,
  getResponseLanguage,
} from '../lib/chatUtils';

// Memoized scroll button component
const ScrollToBottomButton = memo(function ScrollToBottomButton({
  onClick,
  isDark
}: {
  onClick: () => void;
  isDark: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={`absolute bottom-28 left-1/2 -translate-x-1/2 p-2.5 rounded-full shadow-lg transition-all duration-200 z-10
        ${isDark ? 'bg-white/[0.08] hover:bg-white/[0.12] text-white/60 border border-white/[0.06]' : 'bg-white hover:bg-gray-50 text-gray-400 border border-gray-200/80 shadow-sm'}`}
      title="Scroll to bottom"
      aria-label="Scroll to bottom"
    >
      <ArrowDown className="w-4 h-4" />
    </button>
  );
});

export default function Chat() {
  // State
  const [isThinking, setIsThinking] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('auto');
  const [toast, setToast] = useState<{ message: string; type: ToastType } | null>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [regeneratingMessageId, setRegeneratingMessageId] = useState<string | null>(null);
  const [lastResponseMeta, setLastResponseMeta] = useState<{
    model?: string;
    latencyMs?: number;
    citations?: Citation[];
    tokenCount?: number;
  }>({});

  // Store hooks with shallow comparison for better performance
  const {
    activeConversationId,
    messages,
    streamingMessage,
    addMessage,
    replaceLastAssistantMessage,
    setStreamingMessage,
    createConversation
  } = useChatStore(
    (state) => ({
      activeConversationId: state.activeConversationId,
      messages: state.messages,
      streamingMessage: state.streamingMessage,
      addMessage: state.addMessage,
      replaceLastAssistantMessage: state.replaceLastAssistantMessage,
      setStreamingMessage: state.setStreamingMessage,
      createConversation: state.createConversation,
    }),
    shallow
  );

  const { accessToken, isAuthenticated } = useAuthStore(
    (state) => ({ accessToken: state.accessToken, isAuthenticated: state.isAuthenticated }),
    shallow
  );
  const fetchProfile = useProfileStore((state) => state.fetchProfile);
  const resolvedTheme = useThemeStore((state) => state.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  // Refs
  const abortControllerRef = useRef<AbortController | null>(null);

  // Toast helpers
  const showToast = useCallback((message: string, type: ToastType = 'error') => {
    setToast({ message, type });
  }, []);

  const hideToast = useCallback(() => {
    setToast(null);
  }, []);

  // Custom hooks
  const { playingAudioMessageId, handleAudio, stopAudio } = useAudioPlayback({
    selectedLanguage,
    showToast
  });

  const { messagesEndRef, messagesContainerRef, showScrollButton, scrollToBottom } = useChatScroll({
    messagesLength: messages.length,
    streamingMessage
  });

  // Load user profile on mount
  useEffect(() => {
    if (isAuthenticated) {
      fetchProfile();
    }
  }, [isAuthenticated, fetchProfile]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      stopAudio();
    };
  }, [stopAudio]);

  /**
   * Send a chat message with optional file attachments
   */
  const handleSend = async (message: string, files?: File[], language?: string) => {
    if (!message.trim() && !files?.length) return;

    const responseLanguage = getResponseLanguage(language, selectedLanguage);

    // Get or create conversation
    let convId = activeConversationId;
    if (!convId) {
      const conv = createConversation();
      convId = conv.id;
    }

    // Process uploaded files
    let fileContext = '';
    if (files && files.length > 0) {
      const result = await processUploadedFiles(files);
      fileContext = result.context;
      for (const err of result.errors) {
        showToast(err, 'error');
      }
    }

    const fullMessage = fileContext ? `${message}\n${fileContext}` : message;
    const history = buildConversationHistory(messages);

    // Set thinking state FIRST to ensure UI shows loading
    setIsThinking(true);
    setStreamingMessage('');

    // Add user message to store
    const userMessage = createUserMessage(convId, message, files);
    addMessage(userMessage);

    try {
      abortControllerRef.current = new AbortController();

      const endpoint = getChatEndpoint(isAuthenticated);
      const headers = buildChatHeaders(accessToken ?? undefined);
      const body = buildChatRequestBody({
        message: fullMessage,
        conversationId: convId,
        history,
        language: responseLanguage,
        isAuthenticated,
      });

      const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body,
        signal: abortControllerRef.current.signal,
      });

      let fullResponse = '';
      let responseMeta: typeof lastResponseMeta = {};

      const handleStream = async (reader: ReadableStreamDefaultReader<Uint8Array>) => {
        setIsThinking(false);
        fullResponse = await parseSSEStream(
          reader,
          setStreamingMessage,
          (meta) => {
            responseMeta = meta;
            setLastResponseMeta(meta);
          }
        );
      };

      const handleJSONResponse = async (reader: ReadableStreamDefaultReader<Uint8Array>) => {
        setIsThinking(false);
        fullResponse = await parseJSONResponse(
          reader,
          setStreamingMessage,
          (meta) => {
            responseMeta = meta;
            setLastResponseMeta(meta);
          }
        );
      };

      // Handle auth failure with guest fallback
      if (response.status === 401 && isAuthenticated) {
        console.log('Auth token expired, falling back to guest endpoint');
        const guestResponse = await fetch('/api/v2/chat/guest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: fullMessage,
            language: responseLanguage,
          }),
          signal: abortControllerRef.current.signal,
        });
        if (!guestResponse.ok) {
          throw new Error(`HTTP error! status: ${guestResponse.status}`);
        }
        const reader = guestResponse.body?.getReader();
        if (reader) await handleJSONResponse(reader);
      } else if (response.ok) {
        const reader = response.body?.getReader();
        if (reader) {
          if (isAuthenticated) {
            await handleStream(reader);
          } else {
            await handleJSONResponse(reader);
          }
        }
      } else {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setStreamingMessage('');

      const assistantMessage = createAssistantMessage(
        convId,
        fullResponse || 'I apologize, but I was unable to generate a response. Please try again.',
        !fullResponse,
        {
          citations: responseMeta.citations,
          modelUsed: responseMeta.model,
          latencyMs: responseMeta.latencyMs,
          tokenCount: responseMeta.tokenCount,
        }
      );
      addMessage(assistantMessage);

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        setIsThinking(false);
        return;
      }
      console.error('Chat error:', error);
      setStreamingMessage('');
      setIsThinking(false);
      addMessage(createErrorMessage(convId));
    }
  };

  const handleCopy = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const handleRetry = async (messageId?: string) => {
    const currentMessages = useChatStore.getState().messages;

    let userMessageToRetry: Message | undefined;

    if (messageId) {
      const assistantIdx = currentMessages.findIndex(m => m.id === messageId);
      if (assistantIdx > 0) {
        const prevMessage = currentMessages[assistantIdx - 1];
        if (prevMessage?.role === 'user') {
          userMessageToRetry = prevMessage;
        }
      }
    }

    if (!userMessageToRetry) {
      const userMessages = currentMessages.filter(m => m.role === 'user');
      userMessageToRetry = userMessages[userMessages.length - 1];
    }

    if (!userMessageToRetry) return;

    setIsRegenerating(true);
    setRegeneratingMessageId(messageId || null);

    try {
      await handleRegenerateResponse(userMessageToRetry.content, messageId);
    } finally {
      setIsRegenerating(false);
      setRegeneratingMessageId(null);
    }
  };

  const handleRegenerateResponse = async (userMessage: string, replaceMessageId?: string) => {
    const convId = activeConversationId || useChatStore.getState().createConversation().id;

    const currentMessages = useChatStore.getState().messages;
    const history = currentMessages
      .filter(m => m.id !== replaceMessageId)
      .slice(-10)
      .map(m => ({ role: m.role, content: m.content }));

    const langToSend = selectedLanguage === 'auto' ? undefined : selectedLanguage;
    setStreamingMessage('');

    try {
      abortControllerRef.current = new AbortController();

      const endpoint = getChatEndpoint(isAuthenticated);
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: buildChatHeaders(accessToken ?? undefined),
        body: JSON.stringify({
          message: userMessage,
          history,
          language: langToSend,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) throw new Error('Failed to get response');
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      let fullResponse = '';

      if (isAuthenticated) {
        fullResponse = await parseSSEStream(reader, setStreamingMessage);
      } else {
        fullResponse = await parseJSONResponse(reader, setStreamingMessage);
      }

      setStreamingMessage('');

      const newAssistantMessage: Message = {
        id: replaceMessageId || (Date.now() + 1).toString(),
        conversationId: convId,
        role: 'assistant' as const,
        content: fullResponse || 'I apologize, but I was unable to generate a response. Please try again.',
        timestamp: new Date().toISOString(),
        isError: !fullResponse,
      };

      if (replaceMessageId) {
        replaceLastAssistantMessage(newAssistantMessage);
      } else {
        addMessage(newAssistantMessage);
      }

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return;
      console.error('Regenerate error:', error);
      setStreamingMessage('');
      showToast('Failed to regenerate response. Please try again.', 'error');
    }
  };

  const handleError = useCallback((error: string) => {
    showToast(error, 'error');
  }, [showToast]);

  const handleQuickAction = (prompt: string) => {
    handleSend(prompt, undefined, selectedLanguage);
  };

  // Show empty state only when not thinking and no messages/streaming
  const showEmptyState = messages.length === 0 && !streamingMessage && !isThinking;

  return (
    <div className={`flex h-full overflow-hidden ${isDark ? 'bg-[#0a0a0a] text-white' : 'bg-[#fafafa] text-gray-900'}`}>
      {/* Toast Notification */}
      {toast && <Toast message={toast.message} type={toast.type} onClose={hideToast} />}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 w-full relative overflow-hidden">
        {/* Messages Area */}
        <div
          ref={messagesContainerRef}
          className="flex-1 overflow-y-auto overflow-x-hidden scroll-smooth"
        >
          {showEmptyState ? (
            <EmptyState isDark={isDark} onQuickAction={handleQuickAction} />
          ) : (
            <div className="w-full max-w-3xl mx-auto px-4 py-6" id="main-content" role="log" aria-live="polite" aria-label="Chat messages">
              {/* Render all messages */}
              {messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  message={msg}
                  isStreaming={isRegenerating && regeneratingMessageId === msg.id}
                  isPlayingAudio={playingAudioMessageId === msg.id}
                  onCopy={() => handleCopy(msg.content)}
                  onRetry={msg.role === 'assistant' ? (id) => handleRetry(id) : undefined}
                  onAudio={() => handleAudio(msg.content, msg.id)}
                />
              ))}

              {/* Regenerating Indicator */}
              {isRegenerating && streamingMessage && (
                <RegeneratingIndicator isDark={isDark} streamingMessage={streamingMessage} />
              )}

              {/* Thinking Indicator - shown when waiting for response */}
              {isThinking && !streamingMessage && !isRegenerating && (
                <ThinkingIndicator isDark={isDark} />
              )}

              {/* Streaming Message - shown when receiving response */}
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

              <div ref={messagesEndRef} className="h-6" />
            </div>
          )}
        </div>

        {/* Scroll to bottom button - memoized component */}
        {showScrollButton && (
          <ScrollToBottomButton onClick={scrollToBottom} isDark={isDark} />
        )}

        {/* Input Area - Fixed at bottom with proper spacing */}
        <div className={`flex-shrink-0 py-4 px-4
          ${isDark ? 'bg-[#0a0a0a]' : 'bg-[#fafafa]'}`}>
          <div className="w-full max-w-3xl mx-auto">
            <ChatInput
              onSend={handleSend}
              selectedLanguage={selectedLanguage}
              onLanguageChange={setSelectedLanguage}
              disabled={isThinking}
              onError={handleError}
            />
            {/* Subtle footer text */}
            <p className={`text-center text-[11px] mt-3 font-medium tracking-wide
              ${isDark ? 'text-white/20' : 'text-black/25'}`}>
              ShikshaSetu can make mistakes. Verify important information.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
