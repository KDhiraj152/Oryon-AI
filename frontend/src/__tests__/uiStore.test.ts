import { describe, it, expect, beforeEach } from 'vitest';
import { useUIStore } from '../store/uiStore';

describe('uiStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useUIStore.setState({
      theme: 'system',
      resolvedTheme: 'light',
      sidebarOpen: false,
      toasts: [],
      isBusy: false,
    });
  });

  describe('theme', () => {
    it('should default to system theme', () => {
      const state = useUIStore.getState();
      expect(state.theme).toBe('system');
    });

    it('should set theme to dark', () => {
      useUIStore.getState().setTheme('dark');
      const state = useUIStore.getState();
      expect(state.theme).toBe('dark');
      expect(state.resolvedTheme).toBe('dark');
    });

    it('should set theme to light', () => {
      useUIStore.getState().setTheme('light');
      const state = useUIStore.getState();
      expect(state.theme).toBe('light');
      expect(state.resolvedTheme).toBe('light');
    });
  });

  describe('sidebar', () => {
    it('should start with sidebar closed', () => {
      expect(useUIStore.getState().sidebarOpen).toBe(false);
    });

    it('should open sidebar', () => {
      useUIStore.getState().setSidebarOpen(true);
      expect(useUIStore.getState().sidebarOpen).toBe(true);
    });

    it('should toggle sidebar', () => {
      expect(useUIStore.getState().sidebarOpen).toBe(false);
      useUIStore.getState().toggleSidebar();
      expect(useUIStore.getState().sidebarOpen).toBe(true);
      useUIStore.getState().toggleSidebar();
      expect(useUIStore.getState().sidebarOpen).toBe(false);
    });
  });

  describe('toasts', () => {
    it('should add a toast', () => {
      const id = useUIStore.getState().addToast('Test message', 'info');
      expect(id).toBeDefined();
      const toasts = useUIStore.getState().toasts;
      expect(toasts).toHaveLength(1);
      expect(toasts[0].message).toBe('Test message');
      expect(toasts[0].severity).toBe('info');
    });

    it('should remove a toast', () => {
      const id = useUIStore.getState().addToast('Test', 'error', 0);
      expect(useUIStore.getState().toasts).toHaveLength(1);
      useUIStore.getState().removeToast(id);
      expect(useUIStore.getState().toasts).toHaveLength(0);
    });

    it('should clear all toasts', () => {
      useUIStore.getState().addToast('A', 'info', 0);
      useUIStore.getState().addToast('B', 'error', 0);
      expect(useUIStore.getState().toasts).toHaveLength(2);
      useUIStore.getState().clearToasts();
      expect(useUIStore.getState().toasts).toHaveLength(0);
    });

    it('should keep max 5 toasts', () => {
      for (let i = 0; i < 7; i++) {
        useUIStore.getState().addToast(`Toast ${i}`, 'info', 0);
      }
      expect(useUIStore.getState().toasts.length).toBeLessThanOrEqual(5);
    });
  });

  describe('busy state', () => {
    it('should toggle busy state', () => {
      expect(useUIStore.getState().isBusy).toBe(false);
      useUIStore.getState().setBusy(true);
      expect(useUIStore.getState().isBusy).toBe(true);
      useUIStore.getState().setBusy(false);
      expect(useUIStore.getState().isBusy).toBe(false);
    });
  });
});
