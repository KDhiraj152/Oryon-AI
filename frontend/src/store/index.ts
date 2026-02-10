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

// Re-export shallow for use in components
export { shallow } from 'zustand/shallow';

// Auth Store
export interface User {
  id: string;
  name: string;
  email: string;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  accessToken: string | null;
  refreshToken: string | null;
  login: (user: User, accessToken: string, refreshToken: string) => void;
  logout: () => void;
  syncAuthState: () => void;  // Sync state with token storage
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      accessToken: null,
      refreshToken: null,

      login: (user, accessToken, refreshToken) => {
        // Use secure token manager for XSS mitigation
        setAccessToken(accessToken);
        setRefreshToken(refreshToken);
        set({ user, isAuthenticated: true, accessToken, refreshToken });
      },

      logout: () => {
        // Use secure token cleanup
        clearTokens();
        set({ user: null, isAuthenticated: false, accessToken: null, refreshToken: null });
      },

      // Sync store with actual token state (call on app init and visibility change)
      syncAuthState: () => {
        const hasValidToken = checkAuth();
        const currentAuth = get().isAuthenticated;

        if (currentAuth && !hasValidToken) {
          // Token expired/cleared but store thinks we're authenticated
          clearTokens();
          set({ user: null, isAuthenticated: false, accessToken: null, refreshToken: null });
        } else if (!currentAuth && hasValidToken) {
          // Have valid token but store says not authenticated - update token
          const token = getAccessToken();
          set({ accessToken: token, isAuthenticated: true });
        }
      },
    }),
    {
      name: 'auth-storage',
      onRehydrateStorage: () => {
        // Called when store is rehydrated from storage
        // Schedule a sync check after rehydration
        return (state: AuthState | undefined) => {
          if (state) {
            // Use setTimeout to ensure this runs after rehydration is complete
            setTimeout(() => state.syncAuthState(), 0);
          }
        };
      },
    }
  )
);

// Set up visibility change listener to sync auth on tab focus
if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      useAuthStore.getState().syncAuthState();
    }
  });
}

// Citation type for RAG responses
export interface Citation {
  id: string;
  title: string;
  excerpt?: string;
  score: number;
  url?: string;
}

// Chat Store
export interface Message {
  id: string;
  conversationId: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  attachments?: { name: string; url: string; type: string; size: number }[];
  isError?: boolean;
  // New fields for enhanced UX
  citations?: Citation[];
  modelUsed?: string;
  latencyMs?: number;
  tokenCount?: number;
}

export interface Conversation {
  id: string;
  title: string;
  language?: string;
  subject?: string;
  created_at: string;
  updated_at: string;
}

interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  messages: Message[];
  streamingMessage: string;
  createConversation: () => Conversation;
  setActiveConversationId: (id: string | null) => void;
  deleteConversation: (id: string) => void;
  addMessage: (message: Message) => void;
  deleteMessage: (id: string) => void;
  deleteMessagesFrom: (id: string) => void;
  replaceLastAssistantMessage: (message: Message) => void;
  setStreamingMessage: (content: string) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  activeConversationId: null,
  messages: [],
  streamingMessage: '',

  createConversation: () => {
    const newConv: Conversation = {
      id: `conv-${Date.now()}`,
      title: 'New Chat',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    set((state) => ({
      conversations: [newConv, ...state.conversations],
      activeConversationId: newConv.id,
      messages: [],
    }));
    return newConv;
  },

  setActiveConversationId: (id) => {
    set({ activeConversationId: id, messages: [] });
  },

  deleteConversation: (id) => {
    set((state) => ({
      conversations: state.conversations.filter((c) => c.id !== id),
      activeConversationId: state.activeConversationId === id ? null : state.activeConversationId,
      messages: state.activeConversationId === id ? [] : state.messages,
    }));
  },

  addMessage: (message) => {
    set((state) => ({
      messages: [...state.messages, message],
    }));
  },

  deleteMessage: (id) => {
    set((state) => ({
      messages: state.messages.filter((m) => m.id !== id),
    }));
  },

  deleteMessagesFrom: (id) => {
    set((state) => {
      const index = state.messages.findIndex((m) => m.id === id);
      if (index === -1) return state;
      return {
        messages: state.messages.slice(0, index),
      };
    });
  },

  replaceLastAssistantMessage: (message) => {
    set((state) => {
      // Find last assistant message from end (more efficient)
      for (let i = state.messages.length - 1; i >= 0; i--) {
        if (state.messages[i].role === 'assistant') {
          const newMessages = [...state.messages];
          newMessages[i] = message;
          return { messages: newMessages };
        }
      }
      // No assistant message found, add as new
      return { messages: [...state.messages, message] };
    });
  },

  setStreamingMessage: (content) => {
    set({ streamingMessage: content });
  },
}));

// Theme Store
interface ThemeState {
  theme: 'light' | 'dark' | 'system';
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      theme: 'system',
      resolvedTheme: 'light',

      setTheme: (theme) => {
        const root = globalThis.document.documentElement;

        let resolved: 'light' | 'dark';
        if (theme === 'system') {
          resolved = globalThis.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        } else {
          resolved = theme;
        }

        root.classList.remove('light', 'dark');
        root.classList.add(resolved);

        set({ theme, resolvedTheme: resolved });
      },
    }),
    {
      name: 'theme-storage',
    }
  )
);

// Student Profile Store - Simplified (no constraints)
export interface StudentProfile {
  language: string;
}

interface ProfileState {
  profile: StudentProfile | null;
  isLoading: boolean;
  error: string | null;
  fetchProfile: () => Promise<void>;
  updateProfile: (updates: Partial<StudentProfile>) => Promise<void>;
  clearProfile: () => void;
}

const defaultProfile: StudentProfile = {
  language: 'en',
};

export const useProfileStore = create<ProfileState>()(
  persist(
    (set) => ({
      profile: null,
      isLoading: false,
      error: null,

      fetchProfile: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await fetch('/api/v2/profile/me', {
            headers: {
              ...getAuthHeader(),
            },
          });
          if (!response.ok) {
            // Use default profile for guests
            set({ profile: defaultProfile, isLoading: false });
            return;
          }
          const data = await response.json();
          set({
            profile: data.profile,
            isLoading: false,
          });
        } catch {
          set({ profile: defaultProfile, isLoading: false });
        }
      },

      updateProfile: async (updates) => {
        set({ isLoading: true, error: null });
        try {
          const response = await fetch('/api/v2/profile/me', {
            method: 'PUT',
            headers: {
              'Content-Type': 'application/json',
              ...getAuthHeader(),
            },
            body: JSON.stringify(updates),
          });
          if (!response.ok) throw new Error('Failed to update profile');
          const data = await response.json();
          set({ profile: data.profile, isLoading: false });
        } catch (error) {
          set({ error: (error as Error).message, isLoading: false });
        }
      },

      clearProfile: () => {
        set({ profile: null, error: null });
      },
    }),
    {
      name: 'profile-storage',
      partialize: (state) => ({ profile: state.profile }),
    }
  )
);
