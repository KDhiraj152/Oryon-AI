/**
 * MessageSkeleton — Intentional skeleton loader for AI responses.
 *
 * Shown during the "connecting" phase before first token arrives.
 * Feels purposeful — communicates that the system is working.
 * Uses CSS shimmer animation (GPU-accelerated).
 */

import { memo } from 'react';
import { useUIStore } from '../../store/uiStore';
import { OmLogo } from '../landing/OmLogo';

interface MessageSkeletonProps {
  /** Optional thinking text from the model (e.g., "Searching...", "Analyzing...") */
  readonly thinkingText?: string;
}

export const MessageSkeleton = memo(function MessageSkeleton({
  thinkingText,
}: MessageSkeletonProps) {
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  return (
    <div
      className="w-full py-4 animate-message-in"
      role="status"
      aria-label={thinkingText || 'Generating response...'}
      aria-live="polite"
    >
      <div className="w-full max-w-3xl mx-auto px-4">
        <div className="flex gap-4">
          {/* Avatar */}
          <div className={`w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center animate-avatar-pop
            ${isDark ? 'bg-white/[0.06]' : 'bg-gray-100'}`}>
            <OmLogo variant="minimal" size={16} color={isDark ? 'dark' : 'light'} animated />
          </div>

          {/* Skeleton content */}
          <div className="flex-1 min-w-0 pt-0.5">
            {/* Role label */}
            <div className={`text-xs font-medium mb-3 ${isDark ? 'text-white/40' : 'text-gray-400'}`}>
              Oryon
            </div>

            {/* Thinking text or pulsing dots */}
            {thinkingText ? (
              <div className="flex items-center gap-2.5">
                <div className="flex gap-1">
                  <span className={`w-1.5 h-1.5 rounded-full animate-thinking-dot ${isDark ? 'bg-white/40' : 'bg-gray-400'}`} />
                  <span className={`w-1.5 h-1.5 rounded-full animate-thinking-dot ${isDark ? 'bg-white/40' : 'bg-gray-400'}`} />
                  <span className={`w-1.5 h-1.5 rounded-full animate-thinking-dot ${isDark ? 'bg-white/40' : 'bg-gray-400'}`} />
                </div>
                <span className={`text-sm ${isDark ? 'text-white/30' : 'text-gray-400'}`}>
                  {thinkingText}
                </span>
              </div>
            ) : (
              <div className="space-y-3">
                {/* Shimmer bars */}
                <div className={`h-4 rounded-lg animate-shimmer w-[85%] ${isDark ? 'bg-white/[0.06]' : 'bg-gray-100'}`} />
                <div className={`h-4 rounded-lg animate-shimmer w-[65%] [animation-delay:0.15s] ${isDark ? 'bg-white/[0.06]' : 'bg-gray-100'}`} />
                <div className={`h-4 rounded-lg animate-shimmer w-[45%] [animation-delay:0.3s] ${isDark ? 'bg-white/[0.06]' : 'bg-gray-100'}`} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
});
