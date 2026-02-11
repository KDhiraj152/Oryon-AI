# Frontend Architecture

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## Design Philosophy

The frontend is engineered to feel as responsive as a native application while handling the complexity of real-time AI interactions. Students should experience seamless interaction—the underlying system of 6+ AI models should be invisible to them.

React 18 with TypeScript provides the foundation: static typing catches errors before production, and React's mature ecosystem enables rapid development without sacrificing quality.

---

## Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.x | Concurrent rendering, automatic batching, Suspense |
| **TypeScript** | 5.x | Type safety, IDE support, compile-time error detection |
| **Vite** | 5.x | Sub-second HMR, optimized production builds |
| **Tailwind CSS** | 3.x | Utility-first styling, no CSS file management |
| **shadcn/ui** | Latest | Accessible, customizable component primitives |
| **Zustand** | 4.x | Minimal boilerplate state management |
| **React Router** | 6.x | Declarative routing with data loading |

---

## Directory Structure

```
frontend/src/
├── main.tsx                # App entry point (ReactDOM.createRoot)
├── App.tsx                 # Root component with routing & providers
├── index.css               # Global styles (TailwindCSS)
├── vite-env.d.ts           # Vite type declarations
│
├── api/                    # Backend communication layer
│   ├── client.ts           # HTTP client with interceptors
│   ├── index.ts            # API barrel exports
│   ├── types.ts            # Shared API types
│   ├── auth.ts             # Authentication endpoints
│   ├── chat.ts             # Chat & streaming endpoints
│   ├── content.ts          # Content processing endpoints
│   ├── aiCore.ts           # AI engine endpoints
│   ├── audio.ts            # Audio endpoints
│   ├── conversations.ts    # Conversation management
│   ├── progress.ts         # Student progress
│   ├── qa.ts               # Q&A endpoints
│   ├── system.ts           # Health & status endpoints
│   ├── profileReview.ts    # Profile & review endpoints
│   └── v2.ts               # V2 API helpers
│
├── components/             # Reusable UI components
│   ├── ErrorBoundary.tsx   # Error boundary wrapper
│   ├── LightRays.tsx       # Animated light rays effect
│   ├── LogoLoop.tsx        # Animated logo component
│   ├── chat/               # Chat-specific components
│   │   ├── ChatMessage.tsx     # Message bubble rendering
│   │   ├── ChatInput.tsx       # Message input with controls
│   │   ├── ChatIndicators.tsx  # Typing/loading indicators
│   │   ├── EmptyState.tsx      # Empty chat placeholder
│   │   ├── ExportModal.tsx     # Chat export dialog
│   │   ├── Header.tsx          # Chat header bar
│   │   └── Sidebar.tsx         # Conversation list sidebar
│   ├── landing/            # Landing page components
│   │   └── OmLogo.tsx      # Om/Shiksha Setu logo
│   ├── layout/             # Layout components
│   │   └── AppLayout.tsx   # Main app layout with outlet
│   ├── system/             # System status components
│   │   └── SystemStatusCard.tsx
│   └── ui/                 # Base UI primitives
│       ├── Skeleton.tsx    # Loading skeleton
│       ├── Toast.tsx       # Toast notifications
│       └── index.tsx       # UI barrel exports
│
├── context/                # React Context providers
│   ├── SystemStatusContext.tsx  # System health monitoring
│   └── ThemeContext.tsx         # Dark/light theme
│
├── hooks/                  # Custom React hooks
│   └── useChat.ts          # Chat interaction hook
│
├── pages/                  # Top-level route components
│   ├── index.ts            # Page barrel exports
│   ├── LandingPage.tsx     # Landing/home page
│   ├── Auth.tsx            # Authentication page
│   ├── Chat.tsx            # Chat interface
│   └── Settings.tsx        # Settings page
│
├── store/                  # Zustand state stores
│   └── index.ts            # Auth, chat, and settings stores
│
├── lib/                    # Utility libraries
│   └── accessibility.tsx   # Accessibility helpers (SkipLink)
│
└── utils/                  # Utility functions
    └── secureTokens.ts     # XSS-safe token management
```

---

## State Management with Zustand

Zustand was selected over Redux for three reasons:
1. **Minimal boilerplate**: Store definition in 20 lines instead of 100
2. **TypeScript integration**: Full type inference without additional setup
3. **Persistence support**: Built-in localStorage integration

### Authentication Store

The auth store manages user authentication state with secure token handling:

```typescript
interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  accessToken: string | null;
  refreshToken: string | null;
  login: (user: User, accessToken: string, refreshToken: string) => void;
  logout: () => void;
  syncAuthState: () => void;
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
        clearTokens();
        set({ user: null, isAuthenticated: false, accessToken: null, refreshToken: null });
      },

      syncAuthState: () => {
        const hasValidToken = checkAuth();
        const currentAuth = get().isAuthenticated;

        if (currentAuth && !hasValidToken) {
          // Token expired - clear state
          clearTokens();
          set({ user: null, isAuthenticated: false, accessToken: null, refreshToken: null });
        }
      },
    }),
    {
      name: 'auth-storage',
      onRehydrateStorage: () => {
        return (state) => {
          if (state) {
            setTimeout(() => state.syncAuthState(), 0);
          }
        };
      },
    }
  )
);
```

### Chat Store

The chat store handles conversation history with optimistic updates:

```typescript
interface ChatState {
  conversations: Conversation[];
  activeConversation: Conversation | null;
  messages: Message[];
  isStreaming: boolean;
  streamingMessage: string;

  // Actions
  sendMessage: (content: string) => Promise<void>;
  appendStreamChunk: (chunk: string) => void;
  finalizeMessage: (citations?: Citation[]) => void;
  selectConversation: (id: string) => void;
}

export const useChatStore = create<ChatState>()((set, get) => ({
  conversations: [],
  activeConversation: null,
  messages: [],
  isStreaming: false,
  streamingMessage: '',

  sendMessage: async (content) => {
    // Optimistic update - show user message immediately
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    set((state) => ({
      messages: [...state.messages, userMessage],
      isStreaming: true,
      streamingMessage: '',
    }));

    // Start streaming response
    await startStream(content, (chunk) => {
      get().appendStreamChunk(chunk);
    });
  },

  appendStreamChunk: (chunk) => {
    set((state) => ({
      streamingMessage: state.streamingMessage + chunk,
    }));
  },

  finalizeMessage: (citations) => {
    const { streamingMessage } = get();

    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: streamingMessage,
      citations,
      timestamp: new Date(),
    };

    set((state) => ({
      messages: [...state.messages, assistantMessage],
      isStreaming: false,
      streamingMessage: '',
    }));
  },
}));
```

---

## Real-Time Streaming with Server-Sent Events

The `useSSE` hook handles streaming responses from the backend:

```typescript
interface UseSSEOptions {
  url: string;
  onMessage: (data: string) => void;
  onError?: (error: Event) => void;
  onComplete?: () => void;
}

export function useSSE({ url, onMessage, onError, onComplete }: UseSSEOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const connect = useCallback(() => {
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setIsConnected(true);
    };

    eventSource.onmessage = (event) => {
      const data = event.data;

      if (data === '[DONE]') {
        onComplete?.();
        eventSource.close();
        return;
      }

      onMessage(data);
    };

    eventSource.onerror = (error) => {
      setIsConnected(false);
      onError?.(error);

      // Auto-reconnect after 3 seconds
      setTimeout(() => {
        if (eventSourceRef.current?.readyState === EventSource.CLOSED) {
          connect();
        }
      }, 3000);
    };
  }, [url, onMessage, onError, onComplete]);

  const disconnect = useCallback(() => {
    eventSourceRef.current?.close();
    setIsConnected(false);
  }, []);

  return { connect, disconnect, isConnected };
}
```

---

## Audio Recording and Playback

### Audio Recorder Hook

```typescript
interface AudioRecorderState {
  isRecording: boolean;
  audioBlob: Blob | null;
  duration: number;
}

export function useAudioRecorder() {
  const [state, setState] = useState<AudioRecorderState>({
    isRecording: false,
    audioBlob: null,
    duration: 0,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus',
    });

    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
      setState((prev) => ({ ...prev, audioBlob, isRecording: false }));

      // Cleanup stream
      stream.getTracks().forEach((track) => track.stop());
    };

    mediaRecorder.start(100); // Collect data every 100ms
    setState((prev) => ({ ...prev, isRecording: true }));
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
  };

  return {
    ...state,
    startRecording,
    stopRecording,
  };
}
```

### Audio Player Component

```typescript
interface AudioPlayerProps {
  src: string;
  onEnded?: () => void;
}

export function AudioPlayer({ src, onEnded }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateProgress = () => {
      setProgress((audio.currentTime / audio.duration) * 100);
    };

    audio.addEventListener('timeupdate', updateProgress);
    audio.addEventListener('ended', () => {
      setIsPlaying(false);
      onEnded?.();
    });

    return () => {
      audio.removeEventListener('timeupdate', updateProgress);
    };
  }, [onEnded]);

  return (
    <div className="flex items-center gap-2 p-2 bg-secondary rounded-lg">
      <Button variant="ghost" size="icon" onClick={togglePlay}>
        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      </Button>
      <Progress value={progress} className="flex-1" />
      <audio ref={audioRef} src={src} />
    </div>
  );
}
```

---

## API Client with Interceptors

Centralized API client with automatic token refresh:

```typescript
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000,
});

// Request interceptor - add auth header
apiClient.interceptors.request.use((config) => {
  const token = getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor - handle token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = getRefreshToken();
        const response = await axios.post('/api/v2/auth/refresh', {
          refresh_token: refreshToken,
        });

        const { access_token } = response.data;
        setAccessToken(access_token);

        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        return apiClient(originalRequest);
      } catch (refreshError) {
        // Refresh failed - logout user
        useAuthStore.getState().logout();
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);
```

---

## Component Architecture

### Chat Message Component

```typescript
interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={cn(
      "flex gap-3 p-4",
      isUser ? "flex-row-reverse" : "flex-row"
    )}>
      <Avatar className={cn(
        "h-8 w-8",
        isUser ? "bg-primary" : "bg-secondary"
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </Avatar>

      <div className={cn(
        "flex flex-col gap-1 max-w-[80%]",
        isUser ? "items-end" : "items-start"
      )}>
        <Card className={cn(
          "p-3",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        )}>
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {message.content}
          </ReactMarkdown>

          {isStreaming && (
            <span className="inline-block w-2 h-4 bg-current animate-pulse" />
          )}
        </Card>

        {message.citations && message.citations.length > 0 && (
          <Citations sources={message.citations} />
        )}
      </div>
    </div>
  );
}
```

### Chat Input with Voice Support

```typescript
export function ChatInput() {
  const [input, setInput] = useState('');
  const { sendMessage, isStreaming } = useChatStore();
  const { isRecording, audioBlob, startRecording, stopRecording } = useAudioRecorder();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    await sendMessage(input);
    setInput('');
  };

  const handleVoiceInput = async () => {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  };

  // Process recorded audio
  useEffect(() => {
    if (audioBlob) {
      transcribeAudio(audioBlob).then((text) => {
        setInput(text);
      });
    }
  }, [audioBlob]);

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4 border-t">
      <Button
        type="button"
        variant={isRecording ? "destructive" : "outline"}
        size="icon"
        onClick={handleVoiceInput}
      >
        {isRecording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
      </Button>

      <Input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask a question..."
        disabled={isStreaming || isRecording}
        className="flex-1"
      />

      <Button type="submit" disabled={!input.trim() || isStreaming}>
        <Send className="h-4 w-4" />
      </Button>
    </form>
  );
}
```

---

## Performance Optimizations

### Memoization Strategy

```typescript
// Memoize expensive list renders
const MemoizedMessageList = memo(({ messages }: { messages: Message[] }) => (
  <div className="flex flex-col gap-2">
    {messages.map((message) => (
      <ChatMessage key={message.id} message={message} />
    ))}
  </div>
));

// Selective store subscriptions
const messages = useChatStore((state) => state.messages, shallow);
const isStreaming = useChatStore((state) => state.isStreaming);
```

### Code Splitting

```typescript
// Lazy load heavy components
const ChatInterface = lazy(() => import('./pages/ChatInterface'));
const Settings = lazy(() => import('./pages/Settings'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/chat" element={<ChatInterface />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Suspense>
  );
}
```

---

## Responsive Design

The interface adapts to all screen sizes:

```typescript
// Mobile-first responsive layout
<div className="flex flex-col md:flex-row h-screen">
  {/* Sidebar - hidden on mobile, visible on desktop */}
  <aside className="hidden md:flex md:w-64 border-r">
    <ConversationList />
  </aside>

  {/* Main chat area - full width on mobile */}
  <main className="flex-1 flex flex-col">
    <MessageList />
    <ChatInput />
  </main>

  {/* Mobile navigation */}
  <nav className="md:hidden fixed bottom-0 w-full border-t bg-background">
    <MobileNav />
  </nav>
</div>
```

---

*For data flow details, see Section 5: Data Flow.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com
