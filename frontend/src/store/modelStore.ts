/**
 * Model Store — Active model, routing decisions, latency metrics.
 *
 * Owns: current model info, per-request latency tracking, routing log, fallback state.
 * Does NOT own: messages, auth, UI state.
 */

import { create } from 'zustand';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface ModelInfo {
  id: string;
  name: string;
  /** e.g. 'llama-3.1-8b', 'mistral-7b' */
  variant?: string;
  /** Backend load tier: low / medium / high / critical */
  loadTier?: 'low' | 'medium' | 'high' | 'critical';
}

export interface RoutingDecision {
  timestamp: number;
  /** Which model was initially requested. */
  requestedModel: string;
  /** Which model actually served the request. */
  servedBy: string;
  /** Reason for routing (e.g. 'primary', 'fallback-load', 'fallback-error'). */
  reason: 'primary' | 'fallback-load' | 'fallback-error' | 'fallback-timeout' | 'user-selected';
  /** Request latency in ms. */
  latencyMs: number;
}

export interface LatencyMetrics {
  /** Time to first token (TTFT) in ms. */
  ttft: number | null;
  /** Total request duration in ms. */
  totalMs: number | null;
  /** Tokens per second during streaming. */
  tokensPerSecond: number | null;
  /** Smoothed average TTFT over recent requests. */
  avgTtft: number | null;
  /** Smoothed average total latency. */
  avgTotalMs: number | null;
}

interface ModelState {
  // Active model
  activeModel: ModelInfo | null;
  setActiveModel: (model: ModelInfo | null) => void;

  // Available models (from backend)
  availableModels: ModelInfo[];
  setAvailableModels: (models: ModelInfo[]) => void;

  // Backend load
  backendLoad: 'low' | 'medium' | 'high' | 'critical';
  setBackendLoad: (load: 'low' | 'medium' | 'high' | 'critical') => void;

  // Routing decisions (last N)
  routingLog: RoutingDecision[];
  addRoutingDecision: (decision: RoutingDecision) => void;
  clearRoutingLog: () => void;

  // Latency metrics (exponential moving average)
  latency: LatencyMetrics;
  recordLatency: (ttft: number | null, totalMs: number, tokensPerSecond: number | null) => void;
  resetLatency: () => void;

  // Fallback state
  isFallback: boolean;
  fallbackReason: string | null;
  setFallback: (isFallback: boolean, reason?: string) => void;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const MAX_ROUTING_LOG = 20;
const EMA_ALPHA = 0.3; // Weight for new observations in exponential moving average

// ─── Helpers ─────────────────────────────────────────────────────────────────

function ema(prev: number | null, next: number, alpha: number): number {
  if (prev === null) return next;
  return alpha * next + (1 - alpha) * prev;
}

// ─── Store ───────────────────────────────────────────────────────────────────

const initialLatency: LatencyMetrics = {
  ttft: null,
  totalMs: null,
  tokensPerSecond: null,
  avgTtft: null,
  avgTotalMs: null,
};

export const useModelStore = create<ModelState>((set) => ({
  // ── Active model ──
  activeModel: null,
  setActiveModel: (model) => set({ activeModel: model }),

  // ── Available models ──
  availableModels: [],
  setAvailableModels: (models) => set({ availableModels: models }),

  // ── Backend load ──
  backendLoad: 'low',
  setBackendLoad: (load) => set({ backendLoad: load }),

  // ── Routing log ──
  routingLog: [],

  addRoutingDecision: (decision) => {
    set((s) => ({
      routingLog: [...s.routingLog.slice(-(MAX_ROUTING_LOG - 1)), decision],
    }));
  },

  clearRoutingLog: () => set({ routingLog: [] }),

  // ── Latency metrics ──
  latency: { ...initialLatency },

  recordLatency: (ttft, totalMs, tokensPerSecond) => {
    set((s) => ({
      latency: {
        ttft,
        totalMs,
        tokensPerSecond,
        avgTtft: ttft !== null ? ema(s.latency.avgTtft, ttft, EMA_ALPHA) : s.latency.avgTtft,
        avgTotalMs: ema(s.latency.avgTotalMs, totalMs, EMA_ALPHA),
      },
    }));
  },

  resetLatency: () => set({ latency: { ...initialLatency } }),

  // ── Fallback ──
  isFallback: false,
  fallbackReason: null,
  setFallback: (isFallback, reason) => set({ isFallback, fallbackReason: reason ?? null }),
}));
