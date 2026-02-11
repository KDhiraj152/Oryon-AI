/**
 * System API - Hardware status, model status, and system health
 *
 * Connects to the optimized backend V2 API for real-time
 * hardware monitoring and AI model status.
 */

import { API_BASE, getAuthHeader } from './client';

// ==================== Types (matching actual backend responses) ====================

export interface DeviceInfo {
  chip: string;
  type: string;
  gpu_cores: number;
  neural_engine_tops: number;
  unified_memory_gb: number;
  performance_cores: number;
  efficiency_cores: number;
}

export interface BatchSizes {
  embedding: number;
  reranking: number;
  translation: number;
  tts: number;
  stt: number;
  llm_inference: number;
  ocr: number;
  classification: number;
  summarization: number;
}

export interface PerformanceConfig {
  mps_memory_fraction: number;
  mlx_memory_fraction: number;
  prefetch_kv_cache: boolean;
  use_metal_fast_math: boolean;
  metal_preallocate: boolean;
  compile_models: boolean;
  persistent_workers: boolean;
  warmup_iterations: number;
  progressive_warmup: boolean;
  use_channels_last: boolean;
  pin_memory: boolean;
  disable_gc_during_inference: boolean;
  p_core_threads: number;
  qos_user_interactive: number;
}

export interface BackendsAvailable {
  mlx: boolean;
  mps: boolean;
  coreml: boolean;
  ane: boolean;
}

export interface Benchmarks {
  embeddings_texts_per_sec: number;
  reranking_ms_per_doc: number;
  llm_tokens_per_sec: number;
  tts_realtime_factor: number;
  stt_realtime_factor: number;
}

export interface HardwareStatus {
  device: DeviceInfo;
  optimization: {
    batch_sizes: BatchSizes;
    performance_config: PerformanceConfig;
    backends_available: BackendsAvailable;
  };
  benchmarks: Benchmarks;
  // Computed fields for convenience
  chip?: string;
  memory?: {
    total_gb: number;
    available_gb?: number;
    used_percent?: number;
  };
}

export interface ModelInfoFromBackend {
  role: string;
  backend: 'mlx' | 'mps' | 'cpu' | 'coreml';
  loaded: boolean;
  memory_gb: number;
}

export interface ModelsSummary {
  total_models: number;
  loaded_models: number;
  total_memory_gb: number;
  available_memory_gb: number;
}

export type HealthStatus = 'healthy' | 'degraded' | 'error';

export interface ModelsStatus {
  models: Record<string, ModelInfoFromBackend>;
  summary: ModelsSummary;
  // Computed fields for convenience
  status?: HealthStatus;
  total_loaded?: number;
  total_available?: number;
}

// Policy status from backend
export interface PolicyStatus {
  mode: 'OPEN' | 'EDUCATION' | 'RESEARCH' | 'RESTRICTED';
  description: string;
  philosophy: string;
  settings: {
    unrestricted_mode: boolean;
    policy_filters: boolean;
    curriculum_enforcement: boolean;
    grade_adaptation: boolean;
    harmful_content_blocking: boolean;
    jailbreak_detection: boolean;
    external_calls: boolean;
    redact_secrets: boolean;
    redact_pii: boolean;
  };
  blocked_categories: string[];
  allowed: string[];
  stats?: {
    checks_performed: number;
    content_blocked: number;
    mode_switches: number;
  };
}

export interface PolicyModeInfo {
  id: string;
  name: string;
  description: string;
  default: boolean;
  settings: {
    unrestricted_mode: boolean;
    policy_filters: boolean;
    curriculum_enforcement: boolean;
    harmful_content_blocking: boolean;
    external_calls?: boolean;
    jailbreak_detection?: boolean;
  };
}

export interface PolicyModesResponse {
  modes: PolicyModeInfo[];
  safety_notice: string;
}

export interface PolicySwitchResult {
  success: boolean;
  message: string;
  old_mode: string;
  new_mode: string;
  description: string;
  settings: {
    unrestricted_mode: boolean;
    policy_filters: boolean;
    curriculum_enforcement: boolean;
    external_calls: boolean;
    harmful_content_blocking: boolean;
  };
}

export interface SystemHealth {
  status: HealthStatus;
  version: string;
  device?: string;
  backends?: {
    mlx: boolean;
    coreml: boolean;
    mps: boolean;
  };
  policy?: {
    mode: string;
    unrestricted: boolean;
    filters_enabled: boolean;
    curriculum_enforcement: boolean;
    harmful_content_blocking: boolean;
  };
}

// Legacy types for compatibility
export interface DeviceCapabilities {
  chip: string;
  gpu_cores: number;
  cpu_cores: { performance: number; efficiency: number };
  unified_memory_gb: number;
  neural_engine_tops: number;
  mps_available: boolean;
  mlx_available: boolean;
  optimal_batch_sizes: {
    embeddings: number;
    reranking: number;
    llm: number;
    tts: number;
    stt: number;
    ocr: number;
  };
}

export interface ModelInfo {
  name: string;
  type: string;
  status: 'loaded' | 'loading' | 'unloaded' | 'error';
  device: 'mps' | 'mlx' | 'cpu';
  memory_mb: number;
  load_time_ms?: number;
  last_used?: string;
  requests_processed?: number;
  avg_latency_ms?: number;
}

export interface CacheStatus {
  status: HealthStatus;
  tiers: {
    l1_memory: {
      size: number;
      max_size: number;
      hit_rate: number;
    };
    l2_redis: {
      connected: boolean;
      size: number;
      hit_rate: number;
    };
    l3_disk: {
      enabled: boolean;
      size_mb: number;
    };
  };
  total_hit_rate: number;
  total_requests: number;
  timestamp: string;
}

export interface BatchMetrics {
  embeddings: {
    texts_per_second: number;
    avg_batch_size: number;
    queue_length: number;
  };
  reranking: {
    docs_per_second: number;
    avg_latency_ms: number;
  };
  llm: {
    tokens_per_second: number;
    active_requests: number;
  };
}

// ==================== System API ====================

export const system = {
  /**
   * Get hardware status including Apple Silicon capabilities
   */
  async getHardwareStatus(): Promise<HardwareStatus> {
    const response = await fetch(`${API_BASE}/hardware/status`, {
      headers: { ...getAuthHeader() },
    });
    if (!response.ok) {
      throw new Error('Failed to get hardware status');
    }
    const data = await response.json();

    // Normalize the response to add convenience fields
    return {
      ...data,
      chip: data.device?.chip || 'Unknown',
      memory: {
        total_gb: data.device?.unified_memory_gb || 0,
        available_gb: data.device?.unified_memory_gb, // Backend doesn't provide this separately
      },
    };
  },

  /**
   * Get AI models status
   */
  async getModelsStatus(): Promise<ModelsStatus> {
    const response = await fetch(`${API_BASE}/models/status`, {
      headers: { ...getAuthHeader() },
    });
    if (!response.ok) {
      throw new Error('Failed to get models status');
    }
    const data = await response.json();

    // Normalize the response
    const loadedCount = data.summary?.loaded_models || 0;
    const totalCount = data.summary?.total_models || 0;

    return {
      ...data,
      status: loadedCount > 0 ? 'healthy' : 'degraded',
      total_loaded: loadedCount,
      total_available: totalCount,
    };
  },

  /**
   * Get cache status across all tiers
   */
  async getCacheStatus(): Promise<CacheStatus> {
    const response = await fetch(`${API_BASE}/cache/status`, {
      headers: { ...getAuthHeader() },
    });
    if (!response.ok) {
      throw new Error('Failed to get cache status');
    }
    return response.json();
  },

  /**
   * Get overall system health
   */
  async getHealth(): Promise<SystemHealth> {
    const response = await fetch(`${API_BASE}/health`, {
      headers: { ...getAuthHeader() },
    });
    if (!response.ok) {
      throw new Error('Failed to get system health');
    }
    return response.json();
  },

  /**
   * Get batch processing metrics
   */
  async getBatchMetrics(): Promise<BatchMetrics> {
    const response = await fetch(`${API_BASE}/batch/metrics`, {
      headers: { ...getAuthHeader() },
    });
    if (!response.ok) {
      throw new Error('Failed to get batch metrics');
    }
    return response.json();
  },

  /**
   * Warm up models for faster first inference
   */
  async warmupModels(models?: string[]): Promise<{ success: boolean; warmed_up: string[] }> {
    const response = await fetch(`${API_BASE}/models/warmup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ models }),
    });
    if (!response.ok) {
      throw new Error('Failed to warm up models');
    }
    return response.json();
  },

  /**
   * Get performance benchmarks
   */
  async getBenchmarks(): Promise<{
    embeddings_per_second: number;
    rerank_latency_ms: number;
    llm_tokens_per_second: number;
    tts_realtime_factor: number;
    stt_realtime_factor: number;
    last_benchmark: string;
  }> {
    const response = await fetch(`${API_BASE}/hardware/benchmarks`, {
      headers: { ...getAuthHeader() },
    });
    if (!response.ok) {
      throw new Error('Failed to get benchmarks');
    }
    return response.json();
  },

  /**
   * Get policy status - current AI operating mode
   */
  async getPolicyStatus(): Promise<PolicyStatus> {
    const response = await fetch(`${API_BASE}/policy`, {
      headers: { ...getAuthHeader() },
    });
    if (!response.ok) {
      throw new Error('Failed to get policy status');
    }
    return response.json();
  },

  /**
   * Get all available policy modes
   */
  async getPolicyModes(): Promise<PolicyModesResponse> {
    const response = await fetch(`${API_BASE}/policy/modes`, {
      headers: { ...getAuthHeader() },
    });
    if (!response.ok) {
      throw new Error('Failed to get policy modes');
    }
    return response.json();
  },

  /**
   * Switch to a different policy mode
   * @param mode - Target mode: OPEN, EDUCATION, RESEARCH, or RESTRICTED
   */
  async switchPolicyMode(mode: 'OPEN' | 'EDUCATION' | 'RESEARCH' | 'RESTRICTED'): Promise<PolicySwitchResult> {
    const response = await fetch(`${API_BASE}/policy/mode`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeader()
      },
      body: JSON.stringify({ mode }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to switch policy mode' }));
      throw new Error(error.detail || 'Failed to switch policy mode');
    }
    return response.json();
  },
};

export default system;
