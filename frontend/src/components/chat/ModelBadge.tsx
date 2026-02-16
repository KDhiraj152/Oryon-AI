/**
 * ModelBadge — Shows which model generated a response.
 *
 * Displays model name, fallback indicator, and optional latency.
 * Compact by default; expands on hover to show routing details.
 */

import { memo, useState } from 'react';
import { Cpu, AlertTriangle, ChevronDown, ChevronUp, Zap } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';

interface ModelBadgeProps {
  /** Model identifier (e.g. 'llama-3.1-8b'). */
  readonly modelName?: string;
  /** Whether this response came from a fallback model. */
  readonly isFallback?: boolean;
  /** Response latency in ms. */
  readonly latencyMs?: number;
  /** Token count. */
  readonly tokenCount?: number;
  /** Agent/pipeline stage that produced this. */
  readonly agentId?: string;
}

function formatLatency(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTokenCount(count: number): string {
  if (count < 1000) return `${count}`;
  return `${(count / 1000).toFixed(1)}k`;
}

export const ModelBadge = memo(function ModelBadge({
  modelName,
  isFallback,
  latencyMs,
  tokenCount,
  agentId,
}: ModelBadgeProps) {
  const [expanded, setExpanded] = useState(false);
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  if (!modelName && !latencyMs) return null;

  const displayName = modelName
    ? modelName.replace(/^(models\/|local\/)/, '').split('/').pop() || modelName
    : 'Unknown model';

  return (
    <div className="inline-flex flex-col items-start mt-2">
      {/* Compact badge */}
      <button
        onClick={() => setExpanded((e) => !e)}
        className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium transition-all duration-150
          ${isDark
            ? 'bg-white/[0.04] hover:bg-white/[0.08] text-white/40 hover:text-white/60 border border-white/[0.06]'
            : 'bg-black/[0.03] hover:bg-black/[0.06] text-black/40 hover:text-black/60 border border-black/[0.06]'
          }
          ${isFallback
            ? isDark
              ? 'border-amber-500/20 text-amber-400/60'
              : 'border-amber-500/20 text-amber-600/60'
            : ''
          }`}
        aria-label={`Model: ${displayName}${isFallback ? ' (fallback)' : ''}`}
      >
        {isFallback ? (
          <AlertTriangle className="w-3 h-3 text-amber-500/70" />
        ) : (
          <Cpu className="w-3 h-3 opacity-50" />
        )}

        <span className="truncate max-w-[120px]">{displayName}</span>

        {latencyMs !== undefined && (
          <>
            <span className="opacity-30">·</span>
            <span className="tabular-nums">{formatLatency(latencyMs)}</span>
          </>
        )}

        {expanded ? (
          <ChevronUp className="w-3 h-3 opacity-40" />
        ) : (
          <ChevronDown className="w-3 h-3 opacity-40" />
        )}
      </button>

      {/* Expanded details */}
      {expanded && (
        <div
          className={`mt-1 px-3 py-2 rounded-lg text-[11px] leading-relaxed space-y-1 border
            ${isDark
              ? 'bg-white/[0.03] border-white/[0.06] text-white/50'
              : 'bg-black/[0.02] border-black/[0.06] text-black/50'
            }`}
        >
          {modelName && (
            <div className="flex items-center gap-2">
              <Cpu className="w-3 h-3 opacity-50" />
              <span>Model: <span className="font-medium opacity-80">{modelName}</span></span>
            </div>
          )}

          {latencyMs !== undefined && (
            <div className="flex items-center gap-2">
              <Zap className="w-3 h-3 opacity-50" />
              <span>Latency: <span className="font-mono tabular-nums">{formatLatency(latencyMs)}</span></span>
            </div>
          )}

          {tokenCount !== undefined && (
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 text-center opacity-50 font-mono text-[9px]">T</span>
              <span>Tokens: <span className="font-mono tabular-nums">{formatTokenCount(tokenCount)}</span></span>
            </div>
          )}

          {agentId && (
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 text-center opacity-50 font-mono text-[9px]">A</span>
              <span>Agent: <span className="font-medium opacity-80">{agentId}</span></span>
            </div>
          )}

          {isFallback && (
            <div className={`flex items-center gap-2 ${isDark ? 'text-amber-400/60' : 'text-amber-600/60'}`}>
              <AlertTriangle className="w-3 h-3" />
              <span>Served by fallback model</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
});
