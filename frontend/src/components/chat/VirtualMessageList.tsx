/**
 * VirtualMessageList — Virtualized rendering for long conversations.
 *
 * Only renders messages within the visible viewport + a small overscan buffer.
 * No external dependencies (no react-window / react-virtualized).
 *
 * Performance characteristics:
 *   - O(1) render cost regardless of total message count
 *   - Smooth auto-scroll during streaming
 *   - Maintains scroll position on re-render
 *   - Overscan of 3 items above/below viewport for smooth scrolling
 */

import {
  memo,
  useRef,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from 'react';

// ─── Types ───────────────────────────────────────────────────────────────────

interface VirtualMessageListProps<T> {
  /** Array of items to virtualize. */
  readonly items: T[];
  /** Estimated height per item in pixels. */
  readonly estimatedItemHeight?: number;
  /** Number of items to render outside the visible viewport. */
  readonly overscan?: number;
  /** Key extractor. */
  readonly keyExtractor: (item: T, index: number) => string;
  /** Render function for each item. */
  readonly renderItem: (item: T, index: number) => ReactNode;
  /** Whether to auto-scroll to bottom when new items appear. */
  readonly autoScrollToBottom?: boolean;
  /** Footer content (e.g., streaming message, thinking indicator). */
  readonly footer?: ReactNode;
  /** Additional className for the container. */
  readonly className?: string;
  /** Called when scroll position changes (for scroll button). */
  readonly onScrollChange?: (isNearBottom: boolean) => void;
}

// Threshold in px below which we consider the user "at the bottom"
const BOTTOM_THRESHOLD = 200;

// Below this item count, just render everything (no virtualization overhead)
const VIRTUALIZATION_THRESHOLD = 50;

// ─── Component ───────────────────────────────────────────────────────────────

function VirtualMessageListInner<T>({
  items,
  estimatedItemHeight = 120,
  overscan = 3,
  keyExtractor,
  renderItem,
  autoScrollToBottom = true,
  footer,
  className,
  onScrollChange,
}: VirtualMessageListProps<T>) {
  const containerRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const itemHeights = useRef<Map<string, number>>(new Map());
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);
  const isUserScrolledUp = useRef(false);
  const prevItemCount = useRef(items.length);

  // ── Measure container ──
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerHeight(entry.contentRect.height);
      }
    });
    observer.observe(container);
    setContainerHeight(container.clientHeight);

    return () => observer.disconnect();
  }, []);

  // ── Measure rendered items ──
  const measureItem = useCallback((key: string, el: HTMLElement | null) => {
    if (el) {
      const height = el.getBoundingClientRect().height;
      if (height > 0) {
        itemHeights.current.set(key, height);
      }
    }
  }, []);

  // ── Calculate total height and visible range ──
  const { visibleRange, totalHeight, itemOffsets } = useMemo(() => {
    const offsets: number[] = [];
    let total = 0;

    for (let i = 0; i < items.length; i++) {
      offsets.push(total);
      const key = keyExtractor(items[i], i);
      const height = itemHeights.current.get(key) ?? estimatedItemHeight;
      total += height;
    }

    // Find visible range
    let startIdx = 0;
    let endIdx = items.length - 1;

    for (let i = 0; i < items.length; i++) {
      const key = keyExtractor(items[i], i);
      const h = itemHeights.current.get(key) ?? estimatedItemHeight;
      if (offsets[i] + h >= scrollTop) {
        startIdx = i;
        break;
      }
    }

    for (let i = startIdx; i < items.length; i++) {
      if (offsets[i] > scrollTop + containerHeight) {
        endIdx = i;
        break;
      }
    }

    // Apply overscan
    startIdx = Math.max(0, startIdx - overscan);
    endIdx = Math.min(items.length - 1, endIdx + overscan);

    return {
      visibleRange: [startIdx, endIdx] as const,
      totalHeight: total,
      itemOffsets: offsets,
    };
  }, [items, scrollTop, containerHeight, estimatedItemHeight, overscan, keyExtractor]);

  // ── Scroll handler ──
  const handleScroll = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    setScrollTop(container.scrollTop);

    const { scrollTop: st, scrollHeight, clientHeight } = container;
    const nearBottom = scrollHeight - st - clientHeight < BOTTOM_THRESHOLD;
    isUserScrolledUp.current = !nearBottom;
    onScrollChange?.(nearBottom);
  }, [onScrollChange]);

  // ── Auto-scroll on new items ──
  useEffect(() => {
    if (items.length > prevItemCount.current && autoScrollToBottom && !isUserScrolledUp.current) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
    prevItemCount.current = items.length;
  }, [items.length, autoScrollToBottom]);

  // ── Skip virtualization for short lists ──
  const shouldVirtualize = items.length > VIRTUALIZATION_THRESHOLD;

  if (!shouldVirtualize) {
    return (
      <div
        ref={containerRef}
        className={`flex-1 overflow-y-auto overflow-x-hidden scroll-smooth ${className ?? ''}`}
        onScroll={handleScroll}
      >
        <div className="w-full max-w-3xl mx-auto px-4 py-6" role="log" aria-live="polite" aria-label="Chat messages">
          {items.map((item, idx) => (
            <div key={keyExtractor(item, idx)}>
              {renderItem(item, idx)}
            </div>
          ))}
          {footer}
          <div ref={endRef} className="h-6" />
        </div>
      </div>
    );
  }

  // ── Virtualized rendering ──
  const [startIdx, endIdx] = visibleRange;
  const topPadding = itemOffsets[startIdx] ?? 0;
  const bottomPadding = totalHeight - (itemOffsets[endIdx] ?? 0) - (itemHeights.current.get(keyExtractor(items[endIdx], endIdx)) ?? estimatedItemHeight);

  return (
    <div
      ref={containerRef}
      className={`flex-1 overflow-y-auto overflow-x-hidden scroll-smooth ${className ?? ''}`}
      onScroll={handleScroll}
    >
      <div className="w-full max-w-3xl mx-auto px-4 py-6" role="log" aria-live="polite" aria-label="Chat messages">
        {/* Top spacer — dynamic height required for virtualization */}
        {topPadding > 0 && <div className={`h-[${topPadding}px]`} aria-hidden="true" />}

        {/* Visible items */}
        {items.slice(startIdx, endIdx + 1).map((item, i) => {
          const actualIdx = startIdx + i;
          const key = keyExtractor(item, actualIdx);
          return (
            <div
              key={key}
              ref={(el) => measureItem(key, el)}
            >
              {renderItem(item, actualIdx)}
            </div>
          );
        })}

        {/* Bottom spacer — dynamic height required for virtualization */}
        {bottomPadding > 0 && <div className={`h-[${Math.max(0, bottomPadding)}px]`} aria-hidden="true" />}

        {/* Footer (streaming content, indicators) */}
        {footer}

        <div ref={endRef} className="h-6" />
      </div>
    </div>
  );
}

// Memoized wrapper (generic components can't be directly memo'd)
export const VirtualMessageList = memo(VirtualMessageListInner) as typeof VirtualMessageListInner;
