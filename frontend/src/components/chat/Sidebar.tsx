import { useState, useEffect, useRef } from 'react';
import { Plus, MessageSquare, Trash2, Settings, LogOut, X, Search, AlertTriangle, Menu } from 'lucide-react';
import { useChatStore, useThemeStore, useAuthStore } from '../../store';
import { OmLogo } from '../landing/OmLogo';
import { useNavigate } from 'react-router-dom';

// Style helper functions
function getConversationStyle(isActive: boolean, isDark: boolean): string {
  if (isActive) {
    return isDark ? 'bg-white/10 text-white' : 'bg-gray-100 text-gray-900 shadow-sm';
  }
  return isDark
    ? 'text-white/70 hover:text-white hover:bg-white/5'
    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50';
}

interface SidebarProps {
  readonly isOpen: boolean;
  readonly onClose: () => void;
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const navigate = useNavigate();
  const { conversations, setActiveConversationId, activeConversationId, createConversation, deleteConversation } = useChatStore();
  const { user, logout } = useAuthStore();
  const { resolvedTheme } = useThemeStore();
  const isDark = resolvedTheme === 'dark';

  const [searchQuery, setSearchQuery] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const startXRef = useRef<number>(0);

  // Handle swipe to close
  useEffect(() => {
    const sidebar = sidebarRef.current;
    if (!sidebar || !isOpen) return;

    const handleTouchStart = (e: TouchEvent) => {
      startXRef.current = e.touches[0].clientX;
    };

    const handleTouchMove = (e: TouchEvent) => {
      const currentX = e.touches[0].clientX;
      const diff = startXRef.current - currentX;

      // If swiping left more than 50px, close sidebar
      if (diff > 50) {
        onClose();
      }
    };

    sidebar.addEventListener('touchstart', handleTouchStart);
    sidebar.addEventListener('touchmove', handleTouchMove);

    return () => {
      sidebar.removeEventListener('touchstart', handleTouchStart);
      sidebar.removeEventListener('touchmove', handleTouchMove);
    };
  }, [isOpen, onClose]);

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const handleNewChat = () => {
    createConversation();
    onClose();
  };

  const handleSelectConversation = (convId: string) => {
    setActiveConversationId(convId);
    onClose();
  };

  const handleDeleteClick = (convId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirmId(convId);
  };

  const handleDeleteConfirm = (convId: string) => {
    deleteConversation(convId);
    setDeleteConfirmId(null);
  };

  const handleDeleteCancel = () => {
    setDeleteConfirmId(null);
  };

  const handleLogout = () => {
    logout();
    navigate('/landing');
  };

  // Filter conversations by search
  const filteredConversations = searchQuery.trim()
    ? conversations.filter(conv =>
        conv.title.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : conversations;

  // Group conversations by date
  const groupedConversations = filteredConversations.reduce((acc, conv) => {
    const date = new Date(conv.updated_at);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    let group = 'Older';
    if (diffDays === 0) group = 'Today';
    else if (diffDays === 1) group = 'Yesterday';
    else if (diffDays <= 7) group = 'Last 7 Days';
    else if (diffDays <= 30) group = 'Last 30 Days';

    if (!acc[group]) acc[group] = [];
    acc[group].push(conv);
    return acc;
  }, {} as Record<string, typeof conversations>);

  // Order groups
  const groupOrder = ['Today', 'Yesterday', 'Last 7 Days', 'Last 30 Days', 'Older'];
  const orderedGroups = groupOrder.filter(g => groupedConversations[g]);

  return (
    <>
      {/* Overlay - Mobile only */}
      <div
        className={`fixed inset-0 bg-black/60 z-overlay backdrop-blur-sm transition-opacity duration-300 lg:hidden
          ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Sidebar */}
      <aside
        ref={sidebarRef}
        className={`fixed inset-y-0 left-0 z-modal w-[280px] sm:w-[300px] flex flex-col overflow-hidden
          ${isDark ? 'bg-[#0a0a0a]/95 backdrop-blur-2xl border-r border-white/[0.05]' : 'bg-white/95 backdrop-blur-2xl border-r border-black/[0.05]'}
          shadow-2xl lg:shadow-none
          transform transition-all duration-300 ease-out-expo
          lg:static lg:z-auto
          ${isOpen ? 'translate-x-0 lg:w-[300px] lg:opacity-100' : '-translate-x-full lg:translate-x-0 lg:w-0 lg:opacity-0 lg:border-r-0'}`}
        aria-label="Chat history sidebar"
      >
        {/* Header - Hidden on Desktop to avoid duplication with Main Header */}
        <div className={`flex items-center gap-3 p-4 lg:hidden
          ${isDark ? 'border-b border-white/[0.05]' : 'border-b border-black/[0.05]'}`}
        >
          {/* Close button - Aligned to match Main Header position */}
          <button
            onClick={onClose}
            className={`p-2 rounded-full transition-all duration-200
              ${isDark
                ? 'hover:bg-white/10 text-white/60 hover:text-white'
                : 'hover:bg-black/5 text-black/50 hover:text-black'
              }`}
            aria-label="Close sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-3">
            <OmLogo variant="minimal" size={24} color={isDark ? 'dark' : 'light'} animated={false} />
            <span className={`text-[15px] font-semibold tracking-tight
              ${isDark ? 'text-white' : 'text-gray-900'}`}
            >
              Oryon AI
            </span>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="px-4 py-4">
          <button
            onClick={handleNewChat}
            className={`w-full flex items-center justify-center gap-2 px-4 py-3
              rounded-full text-sm font-medium transition-all duration-200 active:scale-[0.98]
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400 focus-visible:ring-offset-2
              ${isDark
                ? 'bg-white text-black hover:bg-gray-200 shadow-lg shadow-white/5'
                : 'bg-black text-white hover:bg-gray-800 shadow-lg shadow-black/10'
              }`}
          >
            <Plus className="w-4 h-4" aria-hidden="true" />
            New chat
          </button>
        </div>

        {/* Search */}
        <div className="px-4 pb-2">
          <div className={`relative flex items-center rounded-full transition-all duration-200 border
            ${isDark
              ? 'bg-white/[0.03] border-white/[0.05] focus-within:bg-white/[0.06] focus-within:border-white/10'
              : 'bg-black/[0.03] border-black/[0.05] focus-within:bg-black/[0.05] focus-within:border-black/10'
            }`}
          >
            <Search className={`w-3.5 h-3.5 ml-3.5 flex-shrink-0
              ${isDark ? 'text-white/40' : 'text-black/40'}`}
              aria-hidden="true"
            />
            <input
              type="text"
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`w-full bg-transparent border-0 px-3 py-2.5 text-[13px]
                placeholder:text-opacity-50 focus:outline-none
                ${isDark
                  ? 'text-white placeholder:text-white/40 caret-white'
                  : 'text-black placeholder:text-black/40 caret-black'
                }`}
              aria-label="Search conversations"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className={`p-1 mr-1.5 rounded-full transition-colors
                  ${isDark ? 'hover:bg-white/10 text-white/40' : 'hover:bg-black/5 text-black/40'}`}
                aria-label="Clear search"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>

        {/* Conversations List */}
        <nav className="flex-1 overflow-y-auto px-2" aria-label="Conversations">
          {filteredConversations.length === 0 ? (
            <div className={`px-4 py-12 text-center ${isDark ? 'text-white/40' : 'text-gray-400'}`}>
              <MessageSquare className="w-8 h-8 mx-auto mb-3 opacity-40" aria-hidden="true" />
              <p className="text-sm font-medium">
                {searchQuery ? 'No matches found' : 'No conversations yet'}
              </p>
              <p className="text-[11px] mt-1 opacity-70">
                {searchQuery ? 'Try a different search' : 'Start a new chat to begin'}
              </p>
            </div>
          ) : (
            <div className="py-2">
              {orderedGroups.map((group) => {
                const groupId = group.split(/\s+/).join('-');
                return (
                <section key={group} className="mb-5" aria-labelledby={`group-${groupId}`}>
                  <h3
                    id={`group-${groupId}`}
                    className={`px-4 py-2 text-[10px] font-bold uppercase tracking-widest opacity-50
                    ${isDark ? 'text-white' : 'text-black'}`}
                  >
                    {group}
                  </h3>
                  <ul className="space-y-0.5">
                  {groupedConversations[group].map((conv) => {
                    const isActive = activeConversationId === conv.id;
                    const isDeleting = deleteConfirmId === conv.id;

                    return (
                      <li key={conv.id} className="relative">
                        {/* Delete confirmation overlay */}
                        {isDeleting && (
                          <div className={`absolute inset-0 z-10 flex items-center justify-between px-3 py-2 rounded-xl mx-2
                            ${isDark ? 'bg-red-500/20 backdrop-blur-md' : 'bg-red-50 backdrop-blur-md'} animate-fadeIn`}
                          >
                            <div className="flex items-center gap-2">
                              <AlertTriangle className={`w-3.5 h-3.5 ${isDark ? 'text-red-400' : 'text-red-500'}`} />
                              <span className={`text-xs font-medium ${isDark ? 'text-red-300' : 'text-red-600'}`}>
                                Delete?
                              </span>
                            </div>
                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => handleDeleteConfirm(conv.id)}
                                className="px-2 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wide transition-colors bg-red-500 text-white hover:bg-red-600"
                              >
                                Delete
                              </button>
                              <button
                                onClick={handleDeleteCancel}
                                className={`px-2 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wide transition-colors
                                  ${isDark
                                    ? 'bg-white/10 text-white hover:bg-white/20'
                                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                  }`}
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        )}

                        <button
                          onClick={() => handleSelectConversation(conv.id)}
                          className={`w-full group flex items-center gap-3 px-3 py-3 rounded-2xl text-sm
                            transition-all duration-200 mb-0.5
                            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400
                            ${getConversationStyle(isActive, isDark)}
                            ${isDeleting ? 'opacity-0' : ''}`}
                          aria-current={isActive ? 'page' : undefined}
                        >
                          <MessageSquare
                            className={`w-4 h-4 flex-shrink-0 ${isActive ? 'opacity-80' : 'opacity-50'}`}
                            aria-hidden="true"
                          />
                          <span className="flex-1 truncate text-left font-medium">{conv.title}</span>
                          <button
                            onClick={(e) => handleDeleteClick(conv.id, e)}
                            className={`p-2 min-h-[36px] min-w-[36px] flex items-center justify-center
                              rounded-full opacity-0 group-hover:opacity-100 transition-all duration-150
                              focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400
                              ${isDark
                                ? 'hover:bg-white/10 text-white/50 hover:text-white'
                                : 'hover:bg-gray-200 text-gray-400 hover:text-gray-600'
                              }`}
                            aria-label={`Delete conversation: ${conv.title}`}
                          >
                            <Trash2 className="w-4 h-4" aria-hidden="true" />
                          </button>
                        </button>
                      </li>
                    );
                  })}
                  </ul>
                </section>
              )})}
            </div>
          )}
        </nav>

        {/* Footer */}
        <div className={`p-3 ${isDark ? 'border-t border-white/10' : 'border-t border-gray-200'}`}>
          {/* User Card */}
          <div className={`flex items-center gap-3 p-3 rounded-2xl mb-3
            ${isDark ? 'bg-white/5' : 'bg-gray-50'}`}
          >
            <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold flex-shrink-0
              ${isDark ? 'bg-white/10 text-white' : 'bg-gray-200 text-gray-700'}`}
            >
              {user?.name?.charAt(0).toUpperCase() || 'G'}
            </div>
            <div className="flex-1 min-w-0">
              <div className={`text-sm font-medium truncate ${isDark ? 'text-white' : 'text-gray-900'}`}>
                {user?.name || 'Guest User'}
              </div>
              <div className={`text-xs truncate ${isDark ? 'text-white/50' : 'text-gray-500'}`}>
                {user?.email || 'Sign in for more features'}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => navigate('/settings')}
              className={`flex-1 flex items-center justify-center gap-2 min-h-touch px-3 py-2.5
                rounded-full text-xs font-medium transition-all duration-200
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400
                ${isDark
                  ? 'text-white/60 hover:text-white hover:bg-white/10'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
              aria-label="Go to settings"
            >
              <Settings className="w-4 h-4" aria-hidden="true" />
              Settings
            </button>
            <button
              onClick={handleLogout}
              className={`flex-1 flex items-center justify-center gap-2 min-h-touch px-3 py-2.5
                rounded-full text-xs font-medium transition-all duration-200
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400
                ${isDark
                  ? 'text-white/60 hover:text-white hover:bg-white/10'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
              aria-label="Sign out"
            >
              <LogOut className="w-4 h-4" aria-hidden="true" />
              Sign out
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
