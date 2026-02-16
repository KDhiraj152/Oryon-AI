/**
 * StreamingMarkdown — Streaming-safe markdown renderer.
 *
 * Handles partial markdown during token-by-token streaming:
 * - Closes unclosed code fences to prevent layout shift
 * - Closes unclosed bold/italic markers
 * - Buffers incomplete markdown constructs
 * - Prevents flicker by memoizing on content length thresholds
 * - Debounces re-renders during fast token arrival
 *
 * Used as a drop-in replacement for MarkdownRenderer during streaming.
 */

import { memo, useMemo, useRef } from 'react';
import type { Components } from 'react-markdown';

interface StreamingMarkdownProps {
  /** Raw markdown content (may be partial during streaming) */
  readonly content: string;
  /** react-markdown Components overrides */
  readonly components: Components;
  /** Whether content is still streaming */
  readonly isStreaming?: boolean;
}

// ─── Markdown Sanitization for Partial Content ──────────────────────────────

/**
 * Sanitize partial markdown so it renders without layout thrashing.
 * This closes any unclosed constructs that would cause the parser
 * to interpret the rest of the page as part of the construct.
 */
function sanitizePartialMarkdown(content: string): string {
  let result = content;

  // ─ Close unclosed code fences ─
  const fenceMatches = result.match(/```/g);
  if (fenceMatches && fenceMatches.length % 2 !== 0) {
    result += '\n```';
  }

  // ─ Close unclosed inline code ─
  const backtickMatches = result.match(/(?<!`)`(?!`)/g);
  if (backtickMatches && backtickMatches.length % 2 !== 0) {
    result += '`';
  }

  // ─ Close unclosed bold markers ─
  const boldMatches = result.match(/\*\*/g);
  if (boldMatches && boldMatches.length % 2 !== 0) {
    result += '**';
  }

  // ─ Close unclosed italic (single asterisk not part of bold) ─
  // Count single asterisks that aren't part of ** pairs
  const singleStarPattern = /(?<!\*)\*(?!\*)/g;
  const singleStarMatches = result.match(singleStarPattern);
  if (singleStarMatches && singleStarMatches.length % 2 !== 0) {
    result += '*';
  }

  // ─ Close unclosed strikethrough ─
  const tildeMatches = result.match(/~~/g);
  if (tildeMatches && tildeMatches.length % 2 !== 0) {
    result += '~~';
  }

  return result;
}

// ─── Render Threshold ───────────────────────────────────────────────────────

/**
 * During fast streaming, we don't need to re-render on every single token.
 * We render when content grows by at least N characters or on final content.
 * This prevents excessive re-paints while maintaining visual fluidity.
 */
const RENDER_THRESHOLD = 3; // Characters of change before re-render

const StreamingMarkdown = memo(function StreamingMarkdown({
  content,
  components,
  isStreaming = false,
}: StreamingMarkdownProps) {
  const lastRenderedLength = useRef(0);

  // Determine if we should re-render based on content delta
  const shouldUpdate = !isStreaming ||
    content.length - lastRenderedLength.current >= RENDER_THRESHOLD ||
    content.length < lastRenderedLength.current; // Content shortened (e.g., edit)

  // Content to render — sanitized during streaming, raw otherwise
  const renderedContent = useMemo(() => {
    if (!shouldUpdate && isStreaming) {
      // Return previous content (memo will use cached value)
      return content;
    }
    lastRenderedLength.current = content.length;
    return isStreaming ? sanitizePartialMarkdown(content) : content;
  }, [content, isStreaming, shouldUpdate]);

  // Lazy-load the actual markdown renderer
  // This component is already loaded inside Suspense in ChatMessage
  return (
    <StreamingMarkdownInner
      content={renderedContent}
      components={components}
      isStreaming={isStreaming}
    />
  );
}, (prev, next) => {
  // Custom comparison: skip re-render if content delta is below threshold during streaming
  if (prev.isStreaming && next.isStreaming) {
    const delta = Math.abs(next.content.length - prev.content.length);
    if (delta < RENDER_THRESHOLD) return true; // Skip render
  }
  return prev.content === next.content && prev.isStreaming === next.isStreaming;
});

/**
 * Inner renderer — separated so the memo boundary above controls updates.
 */
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const remarkPlugins = [remarkGfm, remarkMath];
const rehypePlugins = [[rehypeKatex, { throwOnError: false, strict: false }] as const];

const StreamingMarkdownInner = memo(function StreamingMarkdownInner({
  content,
  components,
}: {
  content: string;
  components: Components;
  isStreaming?: boolean;
}) {
  return (
    <ReactMarkdown
      remarkPlugins={remarkPlugins}
      // @ts-expect-error rehype-katex typing is slightly off with const tuple
      rehypePlugins={rehypePlugins}
      components={components}
    >
      {content}
    </ReactMarkdown>
  );
});

export default StreamingMarkdown;
