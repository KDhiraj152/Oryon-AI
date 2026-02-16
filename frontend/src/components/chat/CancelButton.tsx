/**
 * CancelButton — Stream cancellation control.
 *
 * Appears during streaming with a smooth animation.
 * Provides clear visual feedback for cancelling / cancelled states.
 */

import { memo } from 'react';
import { Square, Loader2 } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';

interface CancelButtonProps {
  /** Whether a stream is active and can be cancelled. */
  readonly isStreaming: boolean;
  /** Whether a cancel request is in progress. */
  readonly isCancelling: boolean;
  /** Called when the user clicks cancel. */
  readonly onCancel: () => void;
}

export const CancelButton = memo(function CancelButton({
  isStreaming,
  isCancelling,
  onCancel,
}: CancelButtonProps) {
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  if (!isStreaming && !isCancelling) return null;

  return (
    <button
      onClick={onCancel}
      disabled={isCancelling}
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
        transition-all duration-200 animate-in fade-in slide-in-from-bottom-2
        ${isCancelling
          ? isDark
            ? 'bg-white/[0.05] text-white/30 cursor-not-allowed'
            : 'bg-black/[0.03] text-black/30 cursor-not-allowed'
          : isDark
            ? 'bg-white/[0.08] hover:bg-red-500/20 text-white/60 hover:text-red-400 border border-white/[0.08] hover:border-red-500/30'
            : 'bg-black/[0.04] hover:bg-red-50 text-black/50 hover:text-red-600 border border-black/[0.06] hover:border-red-200'
        }`}
      title="Stop generating (Esc)"
      aria-label="Stop generating response"
    >
      {isCancelling ? (
        <>
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>Stopping…</span>
        </>
      ) : (
        <>
          <Square className="w-3 h-3 fill-current" />
          <span>Stop</span>
        </>
      )}
    </button>
  );
});
