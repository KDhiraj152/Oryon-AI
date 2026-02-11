import { createContext, useContext, useEffect, useState, useCallback, useMemo, ReactNode } from 'react';
import { system, HardwareStatus, ModelsStatus, SystemHealth, PolicyStatus, PolicySwitchResult } from '../api/system';

type PolicyMode = 'OPEN' | 'EDUCATION' | 'RESEARCH' | 'RESTRICTED';

interface SystemStatusContextType {
  hardware: HardwareStatus | null;
  models: ModelsStatus | null;
  health: SystemHealth | null;
  policy: PolicyStatus | null;
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  isOnline: boolean;
  // Policy management
  switchPolicyMode: (mode: PolicyMode) => Promise<PolicySwitchResult>;
  isSwitchingPolicy: boolean;
}

const SystemStatusContext = createContext<SystemStatusContextType | null>(null);

/**
 * Provider for system status (hardware, models, health, policy)
 * Polls the backend periodically for updates
 */
export function SystemStatusProvider({ children }: { children: ReactNode }) {
  const [hardware, setHardware] = useState<HardwareStatus | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [policy, setPolicy] = useState<PolicyStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [isSwitchingPolicy, setIsSwitchingPolicy] = useState(false);

  // Online/offline detection
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const fetchStatus = useCallback(async () => {
    if (!isOnline) {
      setError('You are offline');
      return;
    }

    try {
      setError(null);

      // Fetch all statuses in parallel
      const [hwStatus, modelsStatus, healthStatus, policyStatus] = await Promise.allSettled([
        system.getHardwareStatus(),
        system.getModelsStatus(),
        system.getHealth(),
        system.getPolicyStatus(),
      ]);

      if (hwStatus.status === 'fulfilled') {
        setHardware(hwStatus.value);
      }
      if (modelsStatus.status === 'fulfilled') {
        setModels(modelsStatus.value);
      }
      if (healthStatus.status === 'fulfilled') {
        setHealth(healthStatus.value);
      }
      if (policyStatus.status === 'fulfilled') {
        setPolicy(policyStatus.value);
      }

      // Set error only if all failed
      if (
        hwStatus.status === 'rejected' &&
        modelsStatus.status === 'rejected' &&
        healthStatus.status === 'rejected'
      ) {
        setError('Failed to connect to backend');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, [isOnline]);

  // Initial fetch
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Poll every 30 seconds
  useEffect(() => {
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    await fetchStatus();
  }, [fetchStatus]);

  // Switch policy mode
  const switchPolicyMode = useCallback(async (mode: PolicyMode): Promise<PolicySwitchResult> => {
    setIsSwitchingPolicy(true);
    try {
      const result = await system.switchPolicyMode(mode);
      // Refresh policy status after switching
      const newPolicy = await system.getPolicyStatus();
      setPolicy(newPolicy);
      return result;
    } finally {
      setIsSwitchingPolicy(false);
    }
  }, []);

  // Memoize context value to prevent unnecessary re-renders of consumers
  const contextValue = useMemo(() => ({
    hardware,
    models,
    health,
    policy,
    isLoading,
    error,
    refresh,
    isOnline,
    switchPolicyMode,
    isSwitchingPolicy,
  }), [hardware, models, health, policy, isLoading, error, refresh, isOnline, switchPolicyMode, isSwitchingPolicy]);

  return (
    <SystemStatusContext.Provider value={contextValue}>
      {children}
    </SystemStatusContext.Provider>
  );
}

/**
 * Hook to access system status
 */
// eslint-disable-next-line react-refresh/only-export-components
export function useSystemStatus(): SystemStatusContextType {
  const context = useContext(SystemStatusContext);
  if (!context) {
    throw new Error('useSystemStatus must be used within a SystemStatusProvider');
  }
  return context;
}

/**
 * Hook to get policy mode
 */
// eslint-disable-next-line react-refresh/only-export-components
export function usePolicyMode() {
  const { policy, health } = useSystemStatus();
  // Try dedicated policy endpoint first, fall back to health
  return policy?.mode || health?.policy?.mode || 'OPEN';
}

/**
 * Hook to get device info
 */
// eslint-disable-next-line react-refresh/only-export-components
export function useDeviceInfo() {
  const { hardware } = useSystemStatus();
  return hardware?.device ?? null;
}

/**
 * Hook to check if specific model is loaded
 */
// eslint-disable-next-line react-refresh/only-export-components
export function useModelStatus(modelName: string) {
  const { models } = useSystemStatus();
  const model = models?.models?.[modelName];
  return model ? { ...model, name: modelName } : null;
}

export default SystemStatusContext;
