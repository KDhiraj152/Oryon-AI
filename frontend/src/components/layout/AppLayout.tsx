import { useState, useEffect, useCallback, memo } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from '../chat/Sidebar';
import Header from '../chat/Header';
import { useChatStore } from '../../store';

// Memoized AppLayout to prevent unnecessary re-renders
const AppLayout = memo(function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const createConversation = useChatStore((state) => state.createConversation);
  const location = useLocation();

  // Set default sidebar state based on screen size
  useEffect(() => {
    const handleResize = () => {
      if (globalThis.innerWidth >= 1024) {
        setSidebarOpen(true);
      } else {
        setSidebarOpen(false);
      }
    };

    // Initial check
    handleResize();
  }, []);

  // Close sidebar on route change (mobile only)
  useEffect(() => {
    if (globalThis.innerWidth < 1024) {
      setSidebarOpen(false);
    }
  }, [location.pathname]);

  const handleNewChat = useCallback(() => {
    createConversation();
    if (globalThis.innerWidth < 1024) {
      setSidebarOpen(false);
    }
  }, [createConversation]);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen(prev => !prev);
  }, []);

  const closeSidebar = useCallback(() => {
    setSidebarOpen(false);
  }, []);

  const isSettings = location.pathname === '/settings';

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--bg-primary)] text-[var(--text-primary)] transition-colors duration-300">
      {/* Sidebar - Handles its own responsive visibility */}
      <Sidebar isOpen={sidebarOpen} onClose={closeSidebar} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 relative transition-all duration-300 ease-out-expo">
        {/* Header - Hidden on Settings page to avoid duplication */}
        {!isSettings && (
          <Header
            onMenuClick={toggleSidebar}
            onNewChat={handleNewChat}
            sidebarOpen={sidebarOpen}
          />
        )}

        {/* Page Content */}
        <main id="main-content" role="main" className="flex-1 overflow-hidden relative">
          {/* Animate page transitions */}
          <div
            key={location.pathname}
            className="h-full w-full animate-enter overflow-y-auto scroll-smooth"
          >
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
});

export default AppLayout;
