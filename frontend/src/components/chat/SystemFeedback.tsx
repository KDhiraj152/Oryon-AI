/**
 * SystemFeedback — Real-time adaptive system feedback during AI processing.
 *
 * Shows contextual status based on what the system is doing:
 *   "Thinking..."          → Initial processing
 *   "Searching knowledge..." → RAG retrieval
 *   "Analyzing context..."   → Context processing
 *   "Generating response..." → Token generation started
 *   "Using fallback model..." → Model fallback occurred
 *
 * Adapts appearance based on elapsed time to set expectations:
 *   < 2s  → Simple thinking dots
 *   2-5s  → Shows "This might take a moment"
 *   > 5s  → Shows elapsed timer
 *
 * GPU-accelerated — only transform/opacity animations.
 */

import { memo, useState, useEffect, useRef } from 'react';
import { useUIStore } from '../../store/uiStore';
import { useModelStore } from '../../store/modelStore';
import { useChatStore } from '../../store/chatStore';
import { useShallow } from 'zustand/react/shallow';

interface SystemFeedbackProps {
  /** Override status text (from stream events) */
  readonly statusText?: string;
}

function useElapsedTime(isActive: boolean): number {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(Date.now());

  useEffect(() => {
    if (!isActive) {
      setElapsed(0);
      return;
    }
    startRef.current = Date.now();
    const tick = () => setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [isActive]);

  return elapsed;
}

function getStatusMessage(elapsed: number, statusText?: string, isFallback?: boolean): string {
  if (isFallback) return 'Switching to fallback model...';
  if (statusText) return statusText;
  if (elapsed > 10) return 'Still working on it...';
  if (elapsed > 5) return 'This is taking a moment...';
  if (elapsed > 2) return 'Generating response...';
  return 'Thinking...';
}

export const SystemFeedback = memo(function SystemFeedback({
  statusText,
}: SystemFeedbackProps) {
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  const { status } = useChatStore(useShallow((s) => ({
    status: s.stream.status,
  })));

  const { isFallback, activeModel } = useModelStore(useShallow((s) => ({
    isFallback: s.isFallback,
    activeModel: s.activeModel,
  })));

  const isActive = status === 'connecting';
  const elapsed = useElapsedTime(isActive);

  if (!isActive) return null;

  const message = getStatusMessage(elapsed, statusText, isFallback);

  return (
    <div
      className="animate-message-in w-full py-3"
      role="status"
      aria-live="polite"
      aria-label={message}
    >
      <div className="w-full max-w-3xl mx-auto px-4">
        <div className="flex items-center gap-3">
          {/* Pulsing dots */}
          <div className="flex gap-1">
            <span className={`w-1.5 h-1.5 rounded-full animate-thinking-dot
              ${isDark ? 'bg-white/40' : 'bg-gray-400'}`} />
            <span className={`w-1.5 h-1.5 rounded-full animate-thinking-dot
              ${isDark ? 'bg-white/40' : 'bg-gray-400'}`} />
            <span className={`w-1.5 h-1.5 rounded-full animate-thinking-dot
              ${isDark ? 'bg-white/40' : 'bg-gray-400'}`} />
          </div>

          {/* Status text */}
          <span className={`text-sm font-medium ${isDark ? 'text-white/30' : 'text-gray-400'}`}>
            {message}
          </span>

          {/* Elapsed timer (appears after 5s) */}
          {elapsed >= 5 && (
            <span className={`text-xs tabular-nums ${isDark ? 'text-white/15' : 'text-gray-300'}`}>
              {elapsed}s
            </span>
          )}

          {/* Model indicator during thinking */}
          {activeModel && (
            <span className={`ml-auto text-[10px] ${isDark ? 'text-white/15' : 'text-gray-300'}`}>
              {activeModel.name}
            </span>
          )}
        </div>

        {/* Progress bar — indeterminate */}
        <div className={`mt-2 h-0.5 rounded-full overflow-hidden ${isDark ? 'bg-white/[0.04]' : 'bg-gray-100'}`}>
          <div className={`h-full w-1/3 rounded-full animate-progress
            ${isDark ? 'bg-white/10' : 'bg-gray-200'}`} />
        </div>
      </div>
    </div>
  );
});
