/**
 * UI Store — Theme, sidebar, toasts, and transient visual state.
 *
 * Owns: theme preference, sidebar open/close, toast queue.
 * Does NOT own: messages, auth, model state.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ─── Types ───────────────────────────────────────────────────────────────────

export type ToastSeverity = 'info' | 'success' | 'warning' | 'error';

export interface ToastItem {
  id: string;
  message: string;
  severity: ToastSeverity;
  /** Auto-dismiss after ms. 0 = manual dismiss only. */
  durationMs: number;
  /** Timestamp for ordering. */
  createdAt: number;
}

interface UIState {
  // Theme
  theme: 'light' | 'dark' | 'system';
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark' | 'system') => void;

  // Sidebar
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;

  // Toast queue (max 5)
  toasts: ToastItem[];
  addToast: (message: string, severity?: ToastSeverity, durationMs?: number) => string;
  removeToast: (id: string) => void;
  clearToasts: () => void;

  // Global busy indicator
  isBusy: boolean;
  setBusy: (busy: boolean) => void;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

let toastId = 0;
function nextToastId(): string {
  return `toast-${++toastId}-${Date.now()}`;
}

function resolveSystemTheme(): 'light' | 'dark' {
  return globalThis.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyThemeToDOM(resolved: 'light' | 'dark') {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  root.classList.remove('light', 'dark');
  root.classList.add(resolved);
}

// ─── Store ───────────────────────────────────────────────────────────────────

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      // ── Theme ──
      theme: 'system',
      resolvedTheme: 'light',

      setTheme: (theme) => {
        const resolved = theme === 'system' ? resolveSystemTheme() : theme;
        applyThemeToDOM(resolved);
        set({ theme, resolvedTheme: resolved });
      },

      // ── Sidebar ──
      sidebarOpen: false,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

      // ── Toasts ──
      toasts: [],

      addToast: (message, severity = 'error', durationMs = 5000) => {
        const id = nextToastId();
        const toast: ToastItem = {
          id,
          message,
          severity,
          durationMs,
          createdAt: Date.now(),
        };

        set((s) => ({
          toasts: [...s.toasts.slice(-4), toast], // Keep max 5
        }));

        // Auto-dismiss
        if (durationMs > 0) {
          setTimeout(() => get().removeToast(id), durationMs);
        }

        return id;
      },

      removeToast: (id) => {
        set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
      },

      clearToasts: () => set({ toasts: [] }),

      // ── Busy ──
      isBusy: false,
      setBusy: (busy) => set({ isBusy: busy }),
    }),
    {
      name: 'ui-storage',
      partialize: (state) => ({
        theme: state.theme,
        resolvedTheme: state.resolvedTheme,
      }),
    }
  )
);

// Listen for system theme changes
if (typeof globalThis !== 'undefined' && globalThis.matchMedia) {
  globalThis.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    const state = useUIStore.getState();
    if (state.theme === 'system') {
      state.setTheme('system');
    }
  });
}

// Apply persisted theme to DOM on initial load
if (typeof document !== 'undefined') {
  const state = useUIStore.getState();
  applyThemeToDOM(state.resolvedTheme);
}
