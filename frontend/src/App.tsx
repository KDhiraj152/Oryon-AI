import { lazy, Suspense, useEffect, memo } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from './store';
import { SystemStatusProvider } from './context/SystemStatusContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import { SkipLink } from './lib/accessibility';
import AppLayout from './components/layout/AppLayout';

// Lazy load pages for better initial load performance
const Chat = lazy(() => import('./pages/ChatV2'));
const Auth = lazy(() => import('./pages/Auth'));
const LandingPage = lazy(() => import('./pages/LandingPage'));
const Settings = lazy(() => import('./pages/Settings'));

// Lightweight loading fallback - no heavy components
const PageLoader = memo(function PageLoader() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--bg-primary)]">
      <div className="w-8 h-8 border-2 border-current border-t-transparent rounded-full animate-spin opacity-30" />
    </div>
  );
});

// OPTIMIZATION: Prefetch components based on current route
const RoutePrefetcher = memo(function RoutePrefetcher() {
  const location = useLocation();

  useEffect(() => {
    // Prefetch likely next routes based on current location
    // Use requestIdleCallback for non-blocking prefetch
    const prefetch = () => {
      if (location.pathname === '/') {
        // On landing page, likely to go to auth or chat
        import('./pages/Auth');
        import('./pages/ChatV2');
      } else if (location.pathname === '/auth') {
        // After auth, likely to go to chat
        import('./pages/ChatV2');
      } else if (location.pathname === '/chat') {
        // In chat, might go to settings
        import('./pages/Settings');
      }
    };

    if ('requestIdleCallback' in globalThis) {
      (globalThis as typeof globalThis & { requestIdleCallback: (cb: () => void) => number }).requestIdleCallback(prefetch);
    } else {
      setTimeout(prefetch, 100);
    }
  }, [location.pathname]);

  return null;
});

// Main App component wrapped in memo
const App = memo(function App() {
  const { isAuthenticated } = useAuthStore();

  return (
    <ErrorBoundary>
      <SystemStatusProvider>
          {/* Skip link for keyboard users */}
          <SkipLink />

          <Router>
            {/* Prefetch routes based on navigation patterns */}
            <RoutePrefetcher />
            <Suspense fallback={<PageLoader />}>
              <Routes>
                {/* Public Landing Page */}
                <Route path="/" element={<LandingPage />} />

                {/* Auth Route */}
                <Route path="/auth" element={<Auth />} />

                {/* Protected Application Routes */}
                <Route element={<AppLayout />}>
                  <Route
                    path="/chat"
                    element={<Chat />}
                  />
                  <Route
                    path="/settings"
                    element={isAuthenticated ? <Settings /> : <Navigate to="/auth" />}
                  />
                </Route>
              </Routes>
            </Suspense>
          </Router>
        </SystemStatusProvider>
    </ErrorBoundary>
  );
});

export default App;
