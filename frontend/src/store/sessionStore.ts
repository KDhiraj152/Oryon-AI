/**
 * Session Store — Authentication & conversation session state.
 *
 * Owns: user identity, tokens, auth lifecycle, visibility-change sync.
 * Does NOT own: messages, UI preferences, model state.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  setAccessToken,
  setRefreshToken,
  clearTokens,
  getAccessToken,
  isAuthenticated as checkAuth,
  getAuthHeader,
} from '../utils/secureTokens';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface User {
  id: string;
  name: string;
  email: string;
}

export interface UserProfile {
  language: string;
}

interface SessionState {
  // Auth
  user: User | null;
  isAuthenticated: boolean;
  accessToken: string | null;
  refreshToken: string | null;
  login: (user: User, accessToken: string, refreshToken: string) => void;
  logout: () => void;
  syncAuthState: () => void;

  // Profile
  profile: UserProfile | null;
  isLoadingProfile: boolean;
  profileError: string | null;
  fetchProfile: () => Promise<void>;
  updateProfile: (updates: Partial<UserProfile>) => Promise<void>;
  clearProfile: () => void;
}

// ─── Defaults ────────────────────────────────────────────────────────────────

const defaultProfile: UserProfile = { language: 'en' };

// ─── Store ───────────────────────────────────────────────────────────────────

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      // ── Auth state ──
      user: null,
      isAuthenticated: false,
      accessToken: null,
      refreshToken: null,

      login: (user, accessToken, refreshToken) => {
        setAccessToken(accessToken);
        setRefreshToken(refreshToken);
        set({ user, isAuthenticated: true, accessToken, refreshToken });
      },

      logout: () => {
        clearTokens();
        set({
          user: null,
          isAuthenticated: false,
          accessToken: null,
          refreshToken: null,
          profile: null,
          profileError: null,
        });
      },

      syncAuthState: () => {
        const hasValidToken = checkAuth();
        const currentAuth = get().isAuthenticated;

        if (currentAuth && !hasValidToken) {
          clearTokens();
          set({ user: null, isAuthenticated: false, accessToken: null, refreshToken: null });
        } else if (!currentAuth && hasValidToken) {
          const token = getAccessToken();
          set({ accessToken: token, isAuthenticated: true });
        }
      },

      // ── Profile state ──
      profile: null,
      isLoadingProfile: false,
      profileError: null,

      fetchProfile: async () => {
        set({ isLoadingProfile: true, profileError: null });
        try {
          const response = await fetch('/api/v2/profile/me', {
            headers: { ...getAuthHeader() },
          });
          if (!response.ok) {
            set({ profile: defaultProfile, isLoadingProfile: false });
            return;
          }
          const data = await response.json();
          set({ profile: data.profile, isLoadingProfile: false });
        } catch {
          set({ profile: defaultProfile, isLoadingProfile: false });
        }
      },

      updateProfile: async (updates) => {
        set({ isLoadingProfile: true, profileError: null });
        try {
          const response = await fetch('/api/v2/profile/me', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
            body: JSON.stringify(updates),
          });
          if (!response.ok) throw new Error('Failed to update profile');
          const data = await response.json();
          set({ profile: data.profile, isLoadingProfile: false });
        } catch (error) {
          set({ profileError: (error as Error).message, isLoadingProfile: false });
        }
      },

      clearProfile: () => set({ profile: null, profileError: null }),
    }),
    {
      name: 'session-storage',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        profile: state.profile,
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          setTimeout(() => state.syncAuthState(), 0);
        }
      },
    }
  )
);

// Sync auth on tab focus
if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      useSessionStore.getState().syncAuthState();
    }
  });
}
