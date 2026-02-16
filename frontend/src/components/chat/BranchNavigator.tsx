/**
 * BranchNavigator — Navigate between conversation branches.
 *
 * When a user edits a message, it creates alternative branches.
 * This component shows "2/3" style navigation (like ChatGPT) allowing
 * users to switch between different versions of a response.
 *
 * Appears inline on messages that have sibling branches.
 */

import { memo, useCallback } from 'react';
import { ChevronLeft, ChevronRight, GitBranch } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';

interface BranchNavigatorProps {
  /** Current branch index (0-based) */
  readonly currentIndex: number;
  /** Total number of branches */
  readonly totalBranches: number;
  /** Called when user navigates to a different branch */
  readonly onNavigate: (index: number) => void;
}

export const BranchNavigator = memo(function BranchNavigator({
  currentIndex,
  totalBranches,
  onNavigate,
}: BranchNavigatorProps) {
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  const canGoBack = currentIndex > 0;
  const canGoForward = currentIndex < totalBranches - 1;

  const goBack = useCallback(() => {
    if (canGoBack) onNavigate(currentIndex - 1);
  }, [canGoBack, currentIndex, onNavigate]);

  const goForward = useCallback(() => {
    if (canGoForward) onNavigate(currentIndex + 1);
  }, [canGoForward, currentIndex, onNavigate]);

  if (totalBranches <= 1) return null;

  const btnClass = `p-1 rounded-md transition-colors duration-150
    ${isDark ? 'hover:bg-white/[0.08]' : 'hover:bg-gray-100'}`;
  const disabledClass = isDark ? 'text-white/10 cursor-not-allowed' : 'text-gray-200 cursor-not-allowed';
  const activeClass = isDark ? 'text-white/50' : 'text-gray-500';

  return (
    <div
      className={`inline-flex items-center gap-0.5 rounded-full px-1 py-0.5
        ${isDark ? 'bg-white/[0.04]' : 'bg-gray-50'}`}
      role="navigation"
      aria-label={`Branch ${currentIndex + 1} of ${totalBranches}`}
    >
      <GitBranch className={`w-3 h-3 mr-0.5 ${isDark ? 'text-white/25' : 'text-gray-400'}`} />

      <button
        onClick={goBack}
        disabled={!canGoBack}
        className={`${btnClass} ${canGoBack ? activeClass : disabledClass}`}
        aria-label="Previous branch"
      >
        <ChevronLeft className="w-3.5 h-3.5" />
      </button>

      <span className={`text-[11px] font-medium tabular-nums min-w-[2rem] text-center
        ${isDark ? 'text-white/40' : 'text-gray-500'}`}>
        {currentIndex + 1}/{totalBranches}
      </span>

      <button
        onClick={goForward}
        disabled={!canGoForward}
        className={`${btnClass} ${canGoForward ? activeClass : disabledClass}`}
        aria-label="Next branch"
      >
        <ChevronRight className="w-3.5 h-3.5" />
      </button>
    </div>
  );
});
