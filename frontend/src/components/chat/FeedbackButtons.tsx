/**
 * FeedbackButtons — Thumbs up/down with optional comment submission.
 *
 * Replaces the static placeholder thumbs buttons in MessageActionButtons.
 * Sends feedback to the store and optionally to the backend.
 */

import { memo, useState, useCallback, useRef, useEffect } from 'react';
import { ThumbsUp, ThumbsDown, Send, X } from 'lucide-react';
import { useChatStore, type MessageFeedback } from '../../store/chatStore';
import { useUIStore } from '../../store/uiStore';

interface FeedbackButtonsProps {
  readonly messageId: string;
  /** Existing feedback on this message */
  readonly feedback?: MessageFeedback;
}

export const FeedbackButtons = memo(function FeedbackButtons({
  messageId,
  feedback,
}: FeedbackButtonsProps) {
  const [showCommentBox, setShowCommentBox] = useState(false);
  const [comment, setComment] = useState('');
  const [pendingRating, setPendingRating] = useState<'positive' | 'negative' | null>(null);
  const commentRef = useRef<HTMLTextAreaElement>(null);

  const setFeedback = useChatStore((s) => s.setFeedback);
  const addToast = useUIStore((s) => s.addToast);
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  const currentRating = feedback?.rating || null;

  // Auto-focus comment textarea
  useEffect(() => {
    if (showCommentBox && commentRef.current) {
      commentRef.current.focus();
    }
  }, [showCommentBox]);

  const handleRate = useCallback((rating: 'positive' | 'negative') => {
    if (currentRating === rating) {
      // Toggle off
      setFeedback(messageId, { rating, timestamp: new Date().toISOString() });
      return;
    }

    if (rating === 'negative') {
      // Show comment box for negative feedback
      setPendingRating('negative');
      setShowCommentBox(true);
    } else {
      // Positive feedback — submit immediately
      setFeedback(messageId, { rating: 'positive', timestamp: new Date().toISOString() });
      addToast('Thanks for the feedback!', 'success', 2000);
    }
  }, [currentRating, messageId, setFeedback, addToast]);

  const submitComment = useCallback(() => {
    if (!pendingRating) return;
    setFeedback(messageId, {
      rating: pendingRating,
      comment: comment.trim() || undefined,
      timestamp: new Date().toISOString(),
    });
    setShowCommentBox(false);
    setComment('');
    setPendingRating(null);
    addToast('Feedback submitted', 'success', 2000);
  }, [pendingRating, comment, messageId, setFeedback, addToast]);

  const cancelComment = useCallback(() => {
    setShowCommentBox(false);
    setComment('');
    setPendingRating(null);
  }, []);

  const baseBtn = `p-2 rounded-full transition-all duration-200 backdrop-blur-sm`;
  const inactiveClass = isDark
    ? 'text-white/30 hover:text-white/70 hover:bg-white/[0.08]'
    : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100';

  return (
    <div className="inline-flex flex-col">
      <div className="flex items-center gap-1">
        {/* Thumbs Up */}
        <button
          onClick={() => handleRate('positive')}
          className={`${baseBtn} ${
            currentRating === 'positive'
              ? 'text-emerald-500 bg-emerald-500/10'
              : inactiveClass
          }`}
          aria-label={currentRating === 'positive' ? 'Undo positive feedback' : 'Good response'}
          title="Good response"
        >
          <ThumbsUp className={`w-4 h-4 ${currentRating === 'positive' ? 'fill-current' : ''}`} />
        </button>

        {/* Thumbs Down */}
        <button
          onClick={() => handleRate('negative')}
          className={`${baseBtn} ${
            currentRating === 'negative'
              ? 'text-red-400 bg-red-500/10'
              : inactiveClass
          }`}
          aria-label={currentRating === 'negative' ? 'Undo negative feedback' : 'Bad response'}
          title="Bad response"
        >
          <ThumbsDown className={`w-4 h-4 ${currentRating === 'negative' ? 'fill-current' : ''}`} />
        </button>
      </div>

      {/* Comment box for negative feedback */}
      {showCommentBox && (
        <div className={`mt-2 animate-scale-up rounded-xl border p-3 max-w-sm
          ${isDark ? 'bg-[#1a1a1a] border-white/10' : 'bg-white border-gray-200 shadow-sm'}`}>
          <p className={`text-xs font-medium mb-2 ${isDark ? 'text-white/60' : 'text-gray-500'}`}>
            What went wrong? (optional)
          </p>
          <textarea
            ref={commentRef}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Help us improve..."
            rows={2}
            className={`w-full resize-none rounded-lg px-3 py-2 text-sm border outline-none transition-colors
              ${isDark
                ? 'bg-white/[0.04] border-white/[0.08] text-white/80 placeholder:text-white/20 focus:border-white/20'
                : 'bg-gray-50 border-gray-200 text-gray-700 placeholder:text-gray-400 focus:border-gray-300'}`}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitComment();
              }
              if (e.key === 'Escape') cancelComment();
            }}
          />
          <div className="flex justify-end gap-2 mt-2">
            <button
              onClick={cancelComment}
              className={`p-1.5 rounded-lg text-xs ${isDark ? 'text-white/40 hover:text-white/60' : 'text-gray-400 hover:text-gray-600'}`}
              aria-label="Cancel feedback"
            >
              <X className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={submitComment}
              className={`p-1.5 rounded-lg text-xs transition-colors
                ${isDark ? 'text-white/60 hover:text-white hover:bg-white/[0.08]' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'}`}
              aria-label="Submit feedback"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
});
