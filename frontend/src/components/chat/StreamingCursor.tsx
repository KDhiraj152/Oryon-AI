/**
 * StreamingCursor — Animated typing cursor shown at the end of streaming text.
 *
 * Provides visual feedback that tokens are being received.
 */

import { memo } from 'react';
import { useUIStore } from '../../store/uiStore';

interface StreamingCursorProps {
  /** Whether the cursor should be visible (during active streaming). */
  readonly visible: boolean;
}

export const StreamingCursor = memo(function StreamingCursor({ visible }: StreamingCursorProps) {
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  if (!visible) return null;

  return (
    <span
      className={`inline-block w-[2px] h-[1.1em] ml-0.5 align-middle animate-pulse rounded-full
        ${isDark ? 'bg-white/60' : 'bg-black/50'}`}
      aria-hidden="true"
    />
  );
});
