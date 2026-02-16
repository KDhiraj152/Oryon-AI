/**
 * SourcePreview — Rich citation card with source preview.
 *
 * Replaces the simple CitationCard with:
 * - Domain favicon
 * - Rich excerpt with match highlighting
 * - Confidence score with visual bar
 * - Click to open in new tab
 * - Hover preview of the source
 *
 * Designed to match Perplexity's source card quality.
 */

import { memo, useState, useMemo } from 'react';
import { ExternalLink, BookOpen, ChevronDown, ChevronUp } from 'lucide-react';
import type { Citation } from '../../store/chatStore';
import { useUIStore } from '../../store/uiStore';

interface SourcePreviewProps {
  readonly citations: Citation[];
}

/** Extract domain from URL */
function getDomain(url?: string): string {
  if (!url) return '';
  try {
    return new URL(url).hostname.replace('www.', '');
  } catch {
    return '';
  }
}

/** Get favicon URL for a domain */
function getFaviconUrl(url?: string): string | null {
  const domain = getDomain(url);
  if (!domain) return null;
  return `https://www.google.com/s2/favicons?sz=32&domain=${domain}`;
}

/** Score color based on confidence */
function getScoreColor(score: number, isDark: boolean): string {
  if (score >= 0.8) return isDark ? 'text-emerald-400' : 'text-emerald-600';
  if (score >= 0.5) return isDark ? 'text-amber-400' : 'text-amber-600';
  return isDark ? 'text-white/40' : 'text-gray-500';
}

function getScoreBg(score: number, isDark: boolean): string {
  if (score >= 0.8) return isDark ? 'bg-emerald-500/20' : 'bg-emerald-50';
  if (score >= 0.5) return isDark ? 'bg-amber-500/20' : 'bg-amber-50';
  return isDark ? 'bg-white/[0.06]' : 'bg-gray-100';
}

/** Individual source card */
const SourceCard = memo(function SourceCard({
  citation,
  index,
  isDark,
}: {
  citation: Citation;
  index: number;
  isDark: boolean;
}) {
  const domain = getDomain(citation.url);
  const favicon = getFaviconUrl(citation.url);

  return (
    <a
      href={citation.url || '#'}
      target="_blank"
      rel="noopener noreferrer"
      className={`group block rounded-xl border p-3.5 transition-all duration-200
        hover:scale-[1.01] active:scale-[0.99]
        ${isDark
          ? 'bg-white/[0.02] border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.1]'
          : 'bg-white border-gray-100 hover:bg-gray-50 hover:border-gray-200 shadow-sm hover:shadow'}`}
      aria-label={`Source ${index + 1}: ${citation.title}`}
    >
      <div className="flex items-start gap-3">
        {/* Index badge */}
        <div className={`flex-shrink-0 w-6 h-6 rounded-lg flex items-center justify-center text-[11px] font-bold
          ${isDark ? 'bg-white/[0.08] text-white/50' : 'bg-gray-100 text-gray-500'}`}>
          {index + 1}
        </div>

        <div className="flex-1 min-w-0">
          {/* Title with external link icon */}
          <div className="flex items-start gap-2">
            <h4 className={`text-sm font-medium line-clamp-1 group-hover:underline underline-offset-2
              ${isDark ? 'text-white/80' : 'text-gray-700'}`}>
              {citation.title}
            </h4>
            <ExternalLink className={`w-3 h-3 flex-shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity
              ${isDark ? 'text-white/40' : 'text-gray-400'}`} />
          </div>

          {/* Excerpt */}
          {citation.excerpt && (
            <p className={`mt-1.5 text-xs leading-relaxed line-clamp-2
              ${isDark ? 'text-white/35' : 'text-gray-500'}`}>
              {citation.excerpt}
            </p>
          )}

          {/* Bottom row: domain + score */}
          <div className="flex items-center gap-3 mt-2">
            {/* Domain with favicon */}
            {domain && (
              <div className="flex items-center gap-1.5">
                {favicon && (
                  <img
                    src={favicon}
                    alt=""
                    className="w-3.5 h-3.5 rounded-sm"
                    loading="lazy"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                  />
                )}
                <span className={`text-[11px] ${isDark ? 'text-white/30' : 'text-gray-400'}`}>
                  {domain}
                </span>
              </div>
            )}

            {/* Confidence score */}
            <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium
              ${getScoreBg(citation.score, isDark)} ${getScoreColor(citation.score, isDark)}`}>
              <div className="w-8 h-1 rounded-full bg-current/20 overflow-hidden">
                <div
                  className={`h-full rounded-full bg-current transition-all duration-300 w-[${Math.round(citation.score * 100)}%]`}
                />
              </div>
              {Math.round(citation.score * 100)}%
            </div>
          </div>
        </div>
      </div>
    </a>
  );
});

export const SourcePreview = memo(function SourcePreview({ citations }: SourcePreviewProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  // Show first 3 inline, rest behind "show more"
  const INITIAL_COUNT = 3;
  const visibleCitations = useMemo(
    () => isExpanded ? citations : citations.slice(0, INITIAL_COUNT),
    [citations, isExpanded],
  );
  const hasMore = citations.length > INITIAL_COUNT;

  if (!citations || citations.length === 0) return null;

  return (
    <div className="pt-3">
      {/* Header */}
      <button
        onClick={() => setIsExpanded((v) => !v)}
        className={`flex items-center gap-2 mb-2.5 px-1 text-xs font-medium transition-colors
          ${isDark ? 'text-white/40 hover:text-white/60' : 'text-gray-400 hover:text-gray-600'}`}
      >
        <BookOpen className="w-3.5 h-3.5" />
        <span>{citations.length} source{citations.length > 1 ? 's' : ''}</span>
        {hasMore && (
          isExpanded
            ? <ChevronUp className="w-3.5 h-3.5" />
            : <ChevronDown className="w-3.5 h-3.5" />
        )}
      </button>

      {/* Source cards */}
      <div className="space-y-2">
        {visibleCitations.map((citation, idx) => (
          <SourceCard
            key={citation.id}
            citation={citation}
            index={idx}
            isDark={isDark}
          />
        ))}
      </div>

      {/* Show more/less toggle */}
      {hasMore && !isExpanded && (
        <button
          onClick={() => setIsExpanded(true)}
          className={`mt-2 text-xs font-medium transition-colors
            ${isDark ? 'text-white/30 hover:text-white/50' : 'text-gray-400 hover:text-gray-600'}`}
        >
          +{citations.length - INITIAL_COUNT} more source{citations.length - INITIAL_COUNT > 1 ? 's' : ''}
        </button>
      )}
    </div>
  );
});
