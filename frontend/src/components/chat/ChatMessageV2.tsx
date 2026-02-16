/**
 * ChatMessage — Production-grade message component.
 *
 * Sub-components extracted into dedicated files for reusability and
 * cognitive simplicity:
 *   chatHelpers.ts        — themed(), childrenToString(), audio helpers
 *   CodeBlock.tsx          — Syntax-highlighted code block with copy
 *   markdownComponents.tsx — react-markdown component overrides
 *   MessageAvatar.tsx      — User/assistant avatar
 *   AttachmentsList.tsx    — File attachment badges
 *   ErrorMessageView.tsx   — Error-state display
 *   useAudioHandler.ts     — Audio toggle hook
 *   MessageActions.tsx     — Action buttons toolbar
 */

import { useState, useMemo, useCallback, memo, lazy, Suspense } from 'react';
import type { Message } from '../../store/chatStore';
import { useUIStore } from '../../store/uiStore';
import { themed } from './chatHelpers';
import { createMarkdownComponents } from './markdownComponents';
import { useAudioHandler } from './useAudioHandler';
import { MessageAvatar } from './MessageAvatar';
import { AttachmentsList } from './AttachmentsList';
import { ErrorMessageView } from './ErrorMessageView';
import { MessageActions } from './MessageActions';
import { ModelBadge } from './ModelBadge';
import { StreamingCursor } from './StreamingCursor';
import { ToolInvocation } from './ToolInvocation';
import { SourcePreview } from './SourcePreview';
import { CollapsibleContent } from './CollapsibleContent';
import { InlineEditor } from './InlineEditor';
import { BranchNavigator } from './BranchNavigator';

// Lazy-load heavy markdown renderer
const LazyStreamingMarkdown = lazy(() => import('./StreamingMarkdown'));

// ─── Interfaces ──────────────────────────────────────────────────────────────

interface ChatMessageProps {
  readonly message: Message;
  readonly isStreaming?: boolean;
  readonly isPlayingAudio?: boolean;
  readonly onRetry?: (messageId: string) => void;
  readonly onCopy?: () => void;
  readonly onAudio?: () => void;
  readonly onEdit?: (messageId: string, newContent: string) => void;
  readonly onBranchNavigate?: (messageId: string, branchIndex: number) => void;
  /** Animation stagger index for entrance animations */
  readonly staggerIndex?: number;
}

// ─── Memo Equality ───────────────────────────────────────────────────────────

function arePropsEqual(prev: ChatMessageProps, next: ChatMessageProps): boolean {
  return (
    prev.message.id === next.message.id &&
    prev.message.content === next.message.content &&
    prev.message.isError === next.message.isError &&
    prev.message.feedback?.rating === next.message.feedback?.rating &&
    prev.message.toolEvents?.length === next.message.toolEvents?.length &&
    prev.isStreaming === next.isStreaming &&
    prev.isPlayingAudio === next.isPlayingAudio &&
    prev.staggerIndex === next.staggerIndex
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────

const ChatMessage = memo(function ChatMessage({
  message,
  isStreaming = false,
  isPlayingAudio = false,
  onRetry,
  onCopy,
  onAudio,
  onEdit,
  onBranchNavigate,
  staggerIndex,
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const isUser = message.role === 'user';
  const isError = message.isError;
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';
  const hasCitations = message.citations && message.citations.length > 0;
  const hasAttachments = message.attachments && message.attachments.length > 0;
  const hasToolEvents = message.toolEvents && message.toolEvents.length > 0;
  const hasBranches = (message.siblingIds?.length ?? 0) > 1;

  const markdownComponents = useMemo(() => createMarkdownComponents(isDark), [isDark]);
  const { isLoadingAudio, handleAudio } = useAudioHandler(onAudio, isPlayingAudio);

  const handleCopy = useCallback(() => {
    onCopy?.();
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [onCopy]);

  const handleStartEdit = useCallback(() => setIsEditing(true), []);
  const handleCancelEdit = useCallback(() => setIsEditing(false), []);
  const handleSaveEdit = useCallback((newContent: string) => {
    setIsEditing(false);
    onEdit?.(message.id, newContent);
  }, [onEdit, message.id]);

  const handleBranchNav = useCallback((index: number) => {
    onBranchNavigate?.(message.id, index);
  }, [onBranchNavigate, message.id]);

  // Error state
  if (isError) {
    return <ErrorMessageView message={message} isDark={isDark} onRetry={onRetry} />;
  }

  const showActions = !isStreaming;
  const isAssistant = !isUser;

  // Stagger animation CSS variable
  const staggerClass = staggerIndex != null
    ? `[--stagger-index:${staggerIndex}]`
    : '';

  return (
    <div
      className={`group w-full py-4 border-b border-transparent
        ${themed(isDark, 'hover:bg-white/[0.015]', 'hover:bg-gray-50/50')}
        ${staggerIndex != null ? 'animate-message-stagger' : 'animate-message-in'}
        ${staggerClass}`}
      role="article"
      aria-label={`${isUser ? 'Your' : 'Oryon'} message`}
    >
      <div className="w-full max-w-3xl mx-auto px-4">
        <div className="flex gap-4">
          {/* Avatar */}
          <MessageAvatar isUser={isUser} isDark={isDark} isStreaming={isStreaming} />

          {/* Content */}
          <div className="flex-1 min-w-0 space-y-0.5 overflow-hidden pt-0.5">
            {/* Role label + branch navigator */}
            <div className="flex items-center gap-2 mb-2">
              <span className={`text-xs font-medium ${themed(isDark, 'text-white/40', 'text-gray-400')}`}>
                {isUser ? 'You' : 'Oryon'}
              </span>
              {message.isEdited && (
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${themed(isDark, 'bg-white/[0.06] text-white/25', 'bg-gray-100 text-gray-400')}`}>
                  edited
                </span>
              )}
              {hasBranches && (
                <BranchNavigator
                  currentIndex={message.branchIndex ?? 0}
                  totalBranches={message.siblingIds?.length ?? 0}
                  onNavigate={handleBranchNav}
                />
              )}
            </div>

            {/* Tool invocations (shown before the response text) */}
            {hasToolEvents && (
              <ToolInvocation events={message.toolEvents!} />
            )}

            {/* Message body — editable for user messages, markdown for assistant */}
            {isEditing && isUser ? (
              <InlineEditor
                content={message.content}
                onSave={handleSaveEdit}
                onCancel={handleCancelEdit}
              />
            ) : (
              <div className={`prose prose-sm max-w-none
                ${isDark ? 'prose-invert' : ''}
                prose-p:leading-7 prose-p:mb-4
                prose-headings:font-semibold
                prose-strong:font-semibold
                prose-ul:my-3 prose-li:my-1`}
              >
                {isUser ? (
                  /* User message — plain text */
                  <p className={`whitespace-pre-wrap m-0 text-[15px] leading-relaxed
                    ${themed(isDark, 'text-white/90', 'text-gray-800')}`}>
                    {message.content}
                  </p>
                ) : (
                  /* Assistant message — streaming-safe markdown */
                  <CollapsibleContent maxHeight={800} defaultCollapsed={!isStreaming}>
                    <Suspense fallback={
                      <p className={`text-[15px] leading-7 ${themed(isDark, 'text-white/90', 'text-gray-700')}`}>
                        {message.content}
                      </p>
                    }>
                      <LazyStreamingMarkdown
                        content={message.content}
                        components={markdownComponents}
                        isStreaming={isStreaming}
                      />
                    </Suspense>
                    <StreamingCursor visible={isStreaming} />
                  </CollapsibleContent>
                )}
              </div>
            )}

            {/* Attachments */}
            {hasAttachments && <AttachmentsList attachments={message.attachments!} isDark={isDark} />}

            {/* Action buttons */}
            {showActions && (
              <MessageActions
                isDark={isDark}
                copied={copied}
                isPlayingAudio={isPlayingAudio}
                isLoadingAudio={isLoadingAudio}
                onCopy={onCopy ? handleCopy : undefined}
                onStartEdit={isUser && onEdit ? handleStartEdit : undefined}
                onRetry={isAssistant && onRetry ? () => onRetry(message.id) : undefined}
                onAudio={isAssistant && onAudio ? handleAudio : undefined}
                messageId={message.id}
                feedback={message.feedback}
              />
            )}

            {/* Citations — rich source preview */}
            {hasCitations && !isStreaming && (
              <SourcePreview citations={message.citations!} />
            )}

            {/* Model badge */}
            {showActions && isAssistant && (message.modelUsed || message.latencyMs) && (
              <div className="pt-2">
                <ModelBadge
                  modelName={message.modelUsed}
                  isFallback={message.isFallback}
                  latencyMs={message.latencyMs}
                  tokenCount={message.tokenCount}
                  agentId={message.agentId}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}, arePropsEqual);

export { ChatMessage };
export default ChatMessage;
