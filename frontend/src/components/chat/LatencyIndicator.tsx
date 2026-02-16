/**
 * LatencyIndicator — Real-time latency metrics bar.
 *
 * Shows TTFT, total latency, and tokens/second as a compact,
 * non-intrusive overlay. Updates live during streaming.
 */

import { memo, useMemo } from 'react';
import { Activity, Zap, Clock, TrendingUp, TrendingDown } from 'lucide-react';
import { useModelStore, type LatencyMetrics } from '../../store/modelStore';
import { useChatStore } from '../../store/chatStore';
import { useUIStore } from '../../store/uiStore';
import { useShallow } from 'zustand/react/shallow';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatMs(ms: number | null): string {
  if (ms === null) return '—';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTps(tps: number | null): string {
  if (tps === null) return '—';
  return `${tps.toFixed(1)} tok/s`;
}

type LatencyTier = 'fast' | 'normal' | 'slow' | 'critical';

function getLatencyTier(ms: number | null): LatencyTier {
  if (ms === null) return 'normal';
  if (ms < 500) return 'fast';
  if (ms < 2000) return 'normal';
  if (ms < 5000) return 'slow';
  return 'critical';
}

function tierColor(tier: LatencyTier, isDark: boolean): string {
  switch (tier) {
    case 'fast':
      return isDark ? 'text-emerald-400/70' : 'text-emerald-600/70';
    case 'normal':
      return isDark ? 'text-white/40' : 'text-black/40';
    case 'slow':
      return isDark ? 'text-amber-400/70' : 'text-amber-600/70';
    case 'critical':
      return isDark ? 'text-red-400/70' : 'text-red-600/70';
  }
}

// ─── Sub-components ──────────────────────────────────────────────────────────

const MetricPill = memo(function MetricPill({
  icon: Icon,
  label,
  value,
  className,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className={`inline-flex items-center gap-1 ${className ?? ''}`} title={label}>
      <Icon className="w-3 h-3 opacity-60" />
      <span className="font-mono tabular-nums text-[10px]">{value}</span>
    </div>
  );
});

// ─── Main Component ──────────────────────────────────────────────────────────

export const LatencyIndicator = memo(function LatencyIndicator() {
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  const latency = useModelStore((s) => s.latency);
  const backendLoad = useModelStore((s) => s.backendLoad);

  const { status, startedAt, firstTokenAt, tokensSoFar } = useChatStore(
    useShallow((s) => ({
      status: s.stream.status,
      startedAt: s.stream.startedAt,
      firstTokenAt: s.stream.firstTokenAt,
      tokensSoFar: s.stream.tokensSoFar,
    }))
  );

  // Live metrics during streaming
  const liveMetrics = useMemo((): LatencyMetrics & { liveTtft: number | null } => {
    if (status !== 'streaming' && status !== 'connecting') {
      return { ...latency, liveTtft: null };
    }

    const now = performance.now();
    const liveTtft = firstTokenAt && startedAt ? firstTokenAt - startedAt : null;
    const elapsed = startedAt ? now - startedAt : 0;
    const liveTps = tokensSoFar > 0 && elapsed > 0 ? (tokensSoFar / elapsed) * 1000 : null;

    return {
      ttft: liveTtft ?? latency.ttft,
      totalMs: elapsed,
      tokensPerSecond: liveTps ?? latency.tokensPerSecond,
      avgTtft: latency.avgTtft,
      avgTotalMs: latency.avgTotalMs,
      liveTtft,
    };
  }, [status, startedAt, firstTokenAt, tokensSoFar, latency]);

  const isActive = status === 'streaming' || status === 'connecting';
  const ttftTier = getLatencyTier(liveMetrics.ttft);
  const ttftColor = tierColor(ttftTier, isDark);

  // Trend indicator for average TTFT
  const isTrendingFaster = liveMetrics.avgTtft !== null && liveMetrics.ttft !== null
    && liveMetrics.ttft < liveMetrics.avgTtft;

  // Don't render when there's nothing to show
  if (!isActive && latency.ttft === null && latency.totalMs === null) {
    return null;
  }

  return (
    <div
      className={`inline-flex items-center gap-3 px-3 py-1 rounded-full text-[10px] font-medium transition-all duration-200
        ${isDark
          ? 'bg-white/[0.03] border border-white/[0.06] text-white/40'
          : 'bg-black/[0.02] border border-black/[0.04] text-black/35'
        }
        ${isActive ? 'animate-pulse' : ''}
      `}
      role="status"
      aria-label="Response latency metrics"
    >
      {/* Live streaming indicator */}
      {isActive && (
        <div className="flex items-center gap-1">
          <div className={`w-1.5 h-1.5 rounded-full ${
            status === 'connecting'
              ? 'bg-amber-400 animate-pulse'
              : 'bg-emerald-400 animate-pulse'
          }`} />
          <span className="text-[10px] opacity-60">
            {status === 'connecting' ? 'Connecting' : 'Streaming'}
          </span>
        </div>
      )}

      {/* TTFT */}
      {liveMetrics.ttft !== null && (
        <MetricPill
          icon={Zap}
          label={`Time to first token: ${formatMs(liveMetrics.ttft)}`}
          value={formatMs(liveMetrics.ttft)}
          className={ttftColor}
        />
      )}

      {/* Total latency */}
      {liveMetrics.totalMs !== null && !isActive && (
        <MetricPill
          icon={Clock}
          label={`Total latency: ${formatMs(liveMetrics.totalMs)}`}
          value={formatMs(liveMetrics.totalMs)}
        />
      )}

      {/* Tokens per second */}
      {liveMetrics.tokensPerSecond !== null && (
        <MetricPill
          icon={Activity}
          label={`Speed: ${formatTps(liveMetrics.tokensPerSecond)}`}
          value={formatTps(liveMetrics.tokensPerSecond)}
        />
      )}

      {/* Trend arrow */}
      {!isActive && liveMetrics.avgTtft !== null && liveMetrics.ttft !== null && (
        <div title={`Avg TTFT: ${formatMs(liveMetrics.avgTtft)}`}>
          {isTrendingFaster ? (
            <TrendingDown className="w-3 h-3 text-emerald-500/50" />
          ) : (
            <TrendingUp className="w-3 h-3 text-amber-500/50" />
          )}
        </div>
      )}

      {/* Backend load */}
      {backendLoad !== 'low' && (
        <div className={`text-[10px] ${
          backendLoad === 'critical'
            ? isDark ? 'text-red-400/60' : 'text-red-600/60'
            : backendLoad === 'high'
              ? isDark ? 'text-amber-400/60' : 'text-amber-600/60'
              : isDark ? 'text-white/30' : 'text-black/30'
        }`}>
          Load: {backendLoad}
        </div>
      )}
    </div>
  );
});
