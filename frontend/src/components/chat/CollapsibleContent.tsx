/**
 * CollapsibleContent — Expand/collapse for long AI outputs.
 *
 * Automatically collapses content beyond a threshold, showing a
 * "Show more" toggle. All transitions are GPU-accelerated (height
 * is animated via max-height, not layout-triggering height).
 */

import { memo, useState, useCallback, useRef, useEffect } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';

interface CollapsibleContentProps {
  /** The content to render */
  readonly children: React.ReactNode;
  /** Collapse threshold in pixels (default: 600) */
  readonly maxHeight?: number;
  /** Whether to initially show collapsed */
  readonly defaultCollapsed?: boolean;
  /** Label for the expand button */
  readonly expandLabel?: string;
}

export const CollapsibleContent = memo(function CollapsibleContent({
  children,
  maxHeight = 600,
  defaultCollapsed = true,
  expandLabel = 'Show more',
}: CollapsibleContentProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const [needsCollapse, setNeedsCollapse] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  // Measure content to determine if collapse is needed
  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setNeedsCollapse(entry.contentRect.height > maxHeight);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [maxHeight]);

  const toggle = useCallback(() => setIsCollapsed((v) => !v), []);
  const ariaProps = isCollapsed
    ? { 'aria-expanded': 'false' as const }
    : { 'aria-expanded': 'true' as const };

  return (
    <div className="relative">
      {/* Content wrapper — animated via max-height (GPU-friendly) */}
      <div
        ref={contentRef}
        className={`transition-[max-height] duration-300 ease-smooth overflow-hidden ${isCollapsed && needsCollapse ? `max-h-[${maxHeight}px]` : ''}`}
      >
        {children}
      </div>

      {/* Gradient fade + toggle button */}
      {needsCollapse && (
        <>
          {isCollapsed && (
            <div
              className={`absolute bottom-8 left-0 right-0 h-20 pointer-events-none
                ${isDark
                  ? 'bg-gradient-to-t from-[#0a0a0a] to-transparent'
                  : 'bg-gradient-to-t from-[#fafafa] to-transparent'}`}
            />
          )}
          <button
            onClick={toggle}
            className={`flex items-center gap-1.5 mx-auto mt-2 px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-200
              ${isDark
                ? 'text-white/40 hover:text-white/60 hover:bg-white/[0.06]'
                : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'}`}
            {...ariaProps}
            aria-label={isCollapsed ? expandLabel : 'Show less'}
          >
            {isCollapsed ? (
              <>
                {expandLabel}
                <ChevronDown className="w-3.5 h-3.5" />
              </>
            ) : (
              <>
                Show less
                <ChevronUp className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </>
      )}
    </div>
  );
});
