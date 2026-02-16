/**
 * useSmartScroll — Intelligent auto-scroll for chat interfaces.
 *
 * Behaviors (matching ChatGPT / Claude UX):
 * 1. Auto-scrolls during streaming — only if user was already at bottom.
 * 2. Stops auto-scrolling if user scrolls up manually (intent to read).
 * 3. Shows "scroll to bottom" button when not at bottom.
 * 4. Resumes auto-scroll when user scrolls back to bottom.
 * 5. Instant-jumps on own message send (always).
 * 6. Uses requestAnimationFrame for 60fps scroll updates during streaming.
 * 7. Passive scroll listeners — no jank.
 * 8. Reduced motion: instant scroll instead of smooth.
 */

import { useRef, useState, useCallback, useEffect } from 'react';
import { prefersReducedMotion } from '../lib/motion';

/** Distance from bottom (px) within which we consider user "at bottom" */
const BOTTOM_THRESHOLD = 120;

/** Debounce for scroll-position checks (ms) */
const SCROLL_DEBOUNCE = 80;

interface UseSmartScrollOptions {
  /** Number of messages — triggers auto-scroll check on change */
  messageCount: number;
  /** Current streaming text — used to drive streaming auto-scroll */
  streamingContent: string;
  /** Whether a stream is currently active */
  isStreaming: boolean;
}

interface UseSmartScrollReturn {
  /** Attach to the scroll container */
  containerRef: React.RefObject<HTMLDivElement>;
  /** Attach to the sentinel at the end of the message list */
  bottomRef: React.RefObject<HTMLDivElement>;
  /** Whether the scroll-to-bottom button should be shown */
  showScrollButton: boolean;
  /** Scroll to bottom (smooth or instant based on reduced motion) */
  scrollToBottom: (instant?: boolean) => void;
  /** Whether auto-scroll is currently locked (user scrolled up) */
  isAutoScrollLocked: boolean;
}

export function useSmartScroll({
  messageCount,
  streamingContent,
  isStreaming,
}: UseSmartScrollOptions): UseSmartScrollReturn {
  const containerRef = useRef<HTMLDivElement>(null!);
  const bottomRef = useRef<HTMLDivElement>(null!);
  const [showScrollButton, setShowScrollButton] = useState(false);

  // Track whether user has manually scrolled away from bottom
  const isUserScrolledUp = useRef(false);
  const lastMessageCount = useRef(messageCount);
  const rafId = useRef<number>(0);
  const scrollDebounceTimer = useRef<ReturnType<typeof setTimeout>>();

  // ── Helpers ──

  const isAtBottom = useCallback((): boolean => {
    const el = containerRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_THRESHOLD;
  }, []);

  const doScroll = useCallback((instant = false) => {
    const el = bottomRef.current;
    if (!el) return;
    const behavior = instant || prefersReducedMotion() ? 'instant' : 'smooth';
    el.scrollIntoView({ behavior, block: 'end' });
  }, []);

  // ── Scroll listener — detect user intent ──

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      clearTimeout(scrollDebounceTimer.current);
      scrollDebounceTimer.current = setTimeout(() => {
        const atBottom = isAtBottom();
        setShowScrollButton(!atBottom && messageCount > 0);

        if (atBottom) {
          // User scrolled back to bottom → resume auto-scroll
          isUserScrolledUp.current = false;
        } else if (isStreaming) {
          // User scrolled up during streaming → lock auto-scroll
          isUserScrolledUp.current = true;
        }
      }, SCROLL_DEBOUNCE);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      container.removeEventListener('scroll', handleScroll);
      clearTimeout(scrollDebounceTimer.current);
    };
  }, [isAtBottom, messageCount, isStreaming]);

  // ── Auto-scroll on new messages (non-streaming) ──

  useEffect(() => {
    const newMessageArrived = messageCount > lastMessageCount.current;
    lastMessageCount.current = messageCount;

    if (newMessageArrived && !isUserScrolledUp.current) {
      // New message added — scroll to it
      doScroll(false);
    }
  }, [messageCount, doScroll]);

  // ── Streaming auto-scroll (RAF-driven for 60fps) ──

  useEffect(() => {
    if (!isStreaming || isUserScrolledUp.current) {
      cancelAnimationFrame(rafId.current);
      return;
    }

    // During streaming, scroll on every frame while content is updating
    const tick = () => {
      if (!isUserScrolledUp.current && isStreaming) {
        const el = containerRef.current;
        if (el) {
          el.scrollTop = el.scrollHeight;
        }
        rafId.current = requestAnimationFrame(tick);
      }
    };
    rafId.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(rafId.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isStreaming, streamingContent]);

  // ── Public scrollToBottom (button click / own message send) ──

  const scrollToBottom = useCallback((instant = false) => {
    isUserScrolledUp.current = false;
    setShowScrollButton(false);
    doScroll(instant);
  }, [doScroll]);

  return {
    containerRef,
    bottomRef,
    showScrollButton,
    scrollToBottom,
    isAutoScrollLocked: isUserScrolledUp.current,
  };
}
