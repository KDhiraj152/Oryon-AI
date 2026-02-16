/**
 * ToolInvocation — Inline tool usage visualization.
 *
 * Shows when the AI calls tools (search, code execution, retrieval, etc.)
 * with real-time status updates:
 *   pending  → spinning icon + tool name
 *   running  → animated progress + elapsed time
 *   success  → checkmark + result preview (collapsible)
 *   error    → error icon + error message
 *
 * Designed for transparent system feedback — users see exactly
 * what the AI is doing, matching the UX of Perplexity/ChatGPT.
 */

import { memo, useState, useCallback } from 'react';
import {
  Search, Code, Database, Globe, FileText,
  CheckCircle2, XCircle, Loader2,
  ChevronDown, ChevronUp, Wrench,
} from 'lucide-react';
import type { ToolEvent } from '../../store/chatStore';
import { useUIStore } from '../../store/uiStore';

interface ToolInvocationProps {
  readonly events: ToolEvent[];
}

/** Map tool names to icons */
function getToolIcon(name: string) {
  const lower = name.toLowerCase();
  if (lower.includes('search') || lower.includes('retrieve')) return Search;
  if (lower.includes('code') || lower.includes('execute') || lower.includes('python')) return Code;
  if (lower.includes('database') || lower.includes('sql') || lower.includes('query')) return Database;
  if (lower.includes('web') || lower.includes('browse') || lower.includes('fetch')) return Globe;
  if (lower.includes('file') || lower.includes('read') || lower.includes('document')) return FileText;
  return Wrench;
}

/** Format tool name for display */
function formatToolName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/^./, (c) => c.toUpperCase());
}

/** Status indicator */
function StatusIcon({ status }: { status: ToolEvent['status'] }) {
  switch (status) {
    case 'pending':
      return <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-400" />;
    case 'running':
      return <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />;
    case 'success':
      return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />;
    case 'error':
      return <XCircle className="w-3.5 h-3.5 text-red-400" />;
  }
}

/** Single tool event card */
const ToolEventCard = memo(function ToolEventCard({
  event,
  isDark,
}: {
  event: ToolEvent;
  isDark: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const Icon = getToolIcon(event.name);
  const hasOutput = event.output && event.output.length > 0;
  const isActive = event.status === 'pending' || event.status === 'running';

  const ariaProps = expanded
    ? { 'aria-expanded': 'true' as const }
    : { 'aria-expanded': 'false' as const };

  const toggleExpand = useCallback(() => {
    if (hasOutput || event.input) setExpanded((v) => !v);
  }, [hasOutput, event.input]);

  return (
    <div
      className={`rounded-xl border transition-all duration-200 overflow-hidden
        ${isActive ? 'animate-fade-in' : ''}
        ${isDark
          ? 'bg-white/[0.02] border-white/[0.06] hover:bg-white/[0.04]'
          : 'bg-gray-50/50 border-gray-100 hover:bg-gray-50'}`}
    >
      {/* Header row */}
      <button
        type="button"
        onClick={toggleExpand}
        className={`w-full flex items-center gap-2.5 px-3.5 py-2.5 text-left transition-colors
          ${hasOutput || event.input ? 'cursor-pointer' : 'cursor-default'}`}
        disabled={!hasOutput && !event.input}
        {...ariaProps}
        aria-label={`Tool: ${formatToolName(event.name)}, status: ${event.status}`}
      >
        {/* Tool icon */}
        <div className={`flex-shrink-0 w-6 h-6 rounded-lg flex items-center justify-center
          ${isDark ? 'bg-white/[0.06]' : 'bg-gray-100'}`}>
          <Icon className={`w-3.5 h-3.5 ${isDark ? 'text-white/50' : 'text-gray-500'}`} />
        </div>

        {/* Tool name + status */}
        <div className="flex-1 min-w-0">
          <span className={`text-xs font-medium ${isDark ? 'text-white/70' : 'text-gray-600'}`}>
            {formatToolName(event.name)}
          </span>
          {event.durationMs != null && event.status === 'success' && (
            <span className={`ml-2 text-[10px] ${isDark ? 'text-white/30' : 'text-gray-400'}`}>
              {event.durationMs}ms
            </span>
          )}
        </div>

        {/* Status */}
        <StatusIcon status={event.status} />

        {/* Expand chevron */}
        {(hasOutput || event.input) && (
          <span className={`flex-shrink-0 ${isDark ? 'text-white/20' : 'text-gray-400'}`}>
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </span>
        )}
      </button>

      {/* Expandable output */}
      {expanded && (
        <div className={`px-3.5 pb-3 animate-slide-down-in`}>
          {event.input && (
            <div className={`mb-2 text-[11px] font-mono rounded-lg px-3 py-2 overflow-x-auto
              ${isDark ? 'bg-white/[0.03] text-white/40' : 'bg-gray-100 text-gray-500'}`}>
              <span className={`text-[10px] uppercase tracking-wider font-semibold block mb-1
                ${isDark ? 'text-white/25' : 'text-gray-400'}`}>
                Input
              </span>
              {JSON.stringify(event.input, null, 2)}
            </div>
          )}
          {hasOutput && (
            <div className={`text-[11px] font-mono rounded-lg px-3 py-2 overflow-x-auto max-h-32 overflow-y-auto
              ${isDark ? 'bg-white/[0.03] text-white/50' : 'bg-gray-100 text-gray-600'}`}>
              <span className={`text-[10px] uppercase tracking-wider font-semibold block mb-1
                ${isDark ? 'text-white/25' : 'text-gray-400'}`}>
                Output
              </span>
              {event.output}
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export const ToolInvocation = memo(function ToolInvocation({ events }: ToolInvocationProps) {
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  if (!events || events.length === 0) return null;

  return (
    <div className="space-y-1.5 my-3" role="status" aria-label="Tool invocations">
      {events.map((event) => (
        <ToolEventCard key={event.id} event={event} isDark={isDark} />
      ))}
    </div>
  );
});
