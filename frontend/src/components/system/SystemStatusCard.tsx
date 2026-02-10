import { memo } from 'react';
import {
  Cpu,
  HardDrive,
  Zap,
  Database,
  Server,
  CheckCircle,
  AlertCircle,
  XCircle,
  RefreshCw,
  Wifi,
  WifiOff
} from 'lucide-react';
import { useSystemStatus } from '../../context/SystemStatusContext';
import { Skeleton } from '../ui/Skeleton';

interface StatusBadgeProps {
  status: 'healthy' | 'degraded' | 'error' | 'loading';
  label?: string;
}

function StatusBadge({ status, label }: StatusBadgeProps) {
  const config = {
    healthy: { icon: CheckCircle, color: 'text-emerald-500', bg: 'bg-emerald-500/10' },
    degraded: { icon: AlertCircle, color: 'text-yellow-500', bg: 'bg-yellow-500/10' },
    error: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-500/10' },
    loading: { icon: RefreshCw, color: 'text-gray-400', bg: 'bg-gray-500/10' },
  };

  const { icon: Icon, color, bg } = config[status];

  return (
    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${bg} ${color}`}>
      <Icon className={`w-3.5 h-3.5 ${status === 'loading' ? 'animate-spin' : ''}`} />
      {label && <span className="capitalize">{label}</span>}
    </div>
  );
}

interface SystemStatusCardProps {
  compact?: boolean;
}

/**
 * System Status Card Component
 * Displays hardware, models, and cache status
 */
export const SystemStatusCard = memo(function SystemStatusCard({ compact = false }: SystemStatusCardProps) {
  const { hardware, models, health, isLoading, error, refresh, isOnline } = useSystemStatus();

  if (!isOnline) {
    return (
      <div className="p-4 rounded-2xl border border-[var(--border-color)] bg-[var(--card-bg)]">
        <div className="flex items-center gap-3 text-yellow-500">
          <WifiOff className="w-5 h-5" />
          <span className="text-sm font-medium">You're offline</span>
        </div>
      </div>
    );
  }

  if (isLoading && !hardware && !models) {
    return (
      <div className="p-4 rounded-2xl border border-[var(--border-color)] bg-[var(--card-bg)] space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton width={120} height={20} />
          <Skeleton width={80} height={24} variant="rectangular" />
        </div>
        <div className="grid grid-cols-2 gap-4">
          {[1, 2, 3, 4].map(i => (
            <Skeleton key={i} height={60} variant="rectangular" />
          ))}
        </div>
      </div>
    );
  }

  if (error && !hardware) {
    return (
      <div className="p-4 rounded-2xl border border-red-500/20 bg-red-500/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 text-red-500">
            <AlertCircle className="w-5 h-5" />
            <span className="text-sm font-medium">Connection Error</span>
          </div>
          <button
            onClick={refresh}
            className="p-2 rounded-lg hover:bg-red-500/10 text-red-500 transition-colors"
            title="Retry connection"
            aria-label="Retry connection"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs text-red-400 mt-2">{error}</p>
      </div>
    );
  }

  if (compact) {
    return (
      <div className="flex items-center gap-3">
        <Wifi className="w-4 h-4 text-emerald-500" />
        <StatusBadge status={health?.status || 'loading'} label={health?.status} />
        {hardware?.chip && (
          <span className="text-xs text-[var(--text-muted)]">{hardware.chip}</span>
        )}
      </div>
    );
  }

  // Extract device info from hardware response
  const deviceInfo = hardware?.device;
  const memoryInfo = hardware?.memory;

  return (
    <div className="p-5 rounded-2xl border border-[var(--border-color)] bg-[var(--card-bg)] space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Server className="w-5 h-5 text-[var(--text-secondary)]" />
          <h3 className="font-semibold text-[var(--text-primary)]">System Status</h3>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={health?.status || 'loading'} label={health?.status} />
          <button
            onClick={refresh}
            disabled={isLoading}
            className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] transition-colors disabled:opacity-50"
            aria-label="Refresh status"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Hardware Info */}
      {hardware && (
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-xl bg-[var(--bg-secondary)]">
            <div className="flex items-center gap-2 text-[var(--text-muted)] mb-1">
              <Cpu className="w-4 h-4" />
              <span className="text-xs font-medium">Chip</span>
            </div>
            <p className="font-semibold text-sm">{hardware.chip || deviceInfo?.chip || 'Unknown'}</p>
          </div>

          <div className="p-3 rounded-xl bg-[var(--bg-secondary)]">
            <div className="flex items-center gap-2 text-[var(--text-muted)] mb-1">
              <HardDrive className="w-4 h-4" />
              <span className="text-xs font-medium">Memory</span>
            </div>
            <p className="font-semibold text-sm">
              {memoryInfo?.total_gb?.toFixed(0) || deviceInfo?.unified_memory_gb?.toFixed(0) || 0} GB
            </p>
          </div>

          <div className="p-3 rounded-xl bg-[var(--bg-secondary)]">
            <div className="flex items-center gap-2 text-[var(--text-muted)] mb-1">
              <Zap className="w-4 h-4" />
              <span className="text-xs font-medium">GPU Cores</span>
            </div>
            <p className="font-semibold text-sm">{deviceInfo?.gpu_cores || 0} cores</p>
          </div>

          <div className="p-3 rounded-xl bg-[var(--bg-secondary)]">
            <div className="flex items-center gap-2 text-[var(--text-muted)] mb-1">
              <Database className="w-4 h-4" />
              <span className="text-xs font-medium">Neural Engine</span>
            </div>
            <p className="font-semibold text-sm">{deviceInfo?.neural_engine_tops || 0} TOPS</p>
          </div>
        </div>
      )}

      {/* Models Status */}
      {models && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
              AI Models
            </span>
            <span className="text-xs text-[var(--text-secondary)]">
              {models.summary?.loaded_models || 0} / {models.summary?.total_models || 0} loaded
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(models.models || {}).slice(0, 6).map(([name, model]) => (
              <div
                key={name}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium ${
                  model.loaded
                    ? 'bg-emerald-500/10 text-emerald-500'
                    : 'bg-gray-500/10 text-gray-400'
                }`}
                title={`${name} - ${model.loaded ? 'loaded' : 'unloaded'} (${model.backend})`}
              >
                {name.split('-')[0]}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer with version */}
      {health && (
        <div className="pt-3 border-t border-[var(--border-color)] flex items-center justify-between text-xs text-[var(--text-muted)]">
          <span>v{health.version}</span>
          <span>Device: {health.device || hardware?.chip || 'Unknown'}</span>
        </div>
      )}
    </div>
  );
});

/**
 * Mini status indicator for header
 */
export const SystemStatusIndicator = memo(function SystemStatusIndicator() {
  const { health, isOnline } = useSystemStatus();

  if (!isOnline) {
    return <WifiOff className="w-4 h-4 text-yellow-500" aria-label="Offline" />;
  }

  const statusColors = {
    healthy: 'bg-emerald-500',
    degraded: 'bg-yellow-500',
    error: 'bg-red-500',
  };

  return (
    <div
      className={`w-2 h-2 rounded-full ${statusColors[health?.status || 'healthy']}`}
      aria-label={`System ${health?.status || 'unknown'}`}
    />
  );
});

export default SystemStatusCard;
