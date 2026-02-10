import { Menu, Plus, Moon, Sun, ChevronDown } from 'lucide-react';
import { useThemeStore } from '../../store';
import { useState } from 'react';
import { useSystemStatus, usePolicyMode } from '../../context/SystemStatusContext';

interface HeaderProps {
  readonly onMenuClick: () => void;
  readonly onNewChat: () => void;
  readonly sidebarOpen?: boolean;
}

export default function Header({ onMenuClick, onNewChat, sidebarOpen = false }: HeaderProps) {
  const { resolvedTheme, setTheme } = useThemeStore();
  const isDark = resolvedTheme === 'dark';
  const [showModelMenu, setShowModelMenu] = useState(false);
  const { health, isOnline } = useSystemStatus();
  const policyMode = usePolicyMode();

  // Policy mode colors and labels - clean, no icons
  const getPolicyDisplay = () => {
    switch (policyMode) {
      case 'OPEN':
        return { label: 'Open', color: 'text-emerald-500', bg: 'bg-emerald-500/10' };
      case 'EDUCATION':
        return { label: 'Education', color: 'text-blue-500', bg: 'bg-blue-500/10' };
      case 'RESEARCH':
        return { label: 'Research', color: 'text-purple-500', bg: 'bg-purple-500/10' };
      case 'RESTRICTED':
        return { label: 'Restricted', color: 'text-amber-500', bg: 'bg-amber-500/10' };
      default:
        return { label: 'Open', color: 'text-emerald-500', bg: 'bg-emerald-500/10' };
    }
  };

  const policyDisplay = getPolicyDisplay();

  // Get connection status - semantic indicator
  // Muted colors - subtle, not attention-seeking
  const getStatusInfo = () => {
    // Browser offline
    if (!isOnline) {
      return { color: 'bg-red-400/70', animate: false, label: 'Offline' };
    }
    // Backend healthy and ready
    if (health?.status === 'healthy') {
      return { color: 'bg-emerald-400/80', animate: false, label: 'Ready' };
    }
    // Backend responded but not healthy (warming up / degraded / error)
    if (health?.status === 'degraded' || health?.status === 'error') {
      return { color: 'bg-amber-400/70', animate: true, label: 'Warming up' };
    }
    // health is null = backend not responding - muted red
    return { color: 'bg-red-400/70', animate: false, label: 'Backend offline' };
  };

  const statusInfo = getStatusInfo();

  return (
    <header
      className={`absolute top-0 left-0 right-0 z-50 h-[60px] flex items-center transition-all duration-200
        ${isDark
          ? 'bg-[#0a0a0a]/80 backdrop-blur-xl border-b border-white/[0.05]'
          : 'bg-white/80 backdrop-blur-xl border-b border-black/[0.05]'}`}
      role="banner"
    >
      <div className="flex items-center justify-between w-full px-4">
        {/* Left side - minimal buttons */}
        <div className="flex items-center gap-1">
          <button
            onClick={onMenuClick}
            className={`p-2 rounded-full transition-all duration-200
              ${isDark
                ? 'hover:bg-white/[0.08] active:bg-white/[0.12] text-white/60 hover:text-white'
                : 'hover:bg-black/[0.05] active:bg-black/[0.08] text-black/50 hover:text-black'}
              ${sidebarOpen ? 'opacity-50' : ''}`}
            aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
          >
            <Menu className="w-5 h-5" />
          </button>

          <button
            onClick={onNewChat}
            className={`p-2 rounded-full transition-all duration-200
              ${isDark
                ? 'hover:bg-white/[0.08] active:bg-white/[0.12] text-white/60 hover:text-white'
                : 'hover:bg-black/[0.05] active:bg-black/[0.08] text-black/50 hover:text-black'}`}
            aria-label="New chat"
            title="New chat"
          >
            <Plus className="w-5 h-5" />
          </button>
        </div>

        {/* Center - Clean model selector like ChatGPT/Gemini */}
        <button
          onClick={() => setShowModelMenu(!showModelMenu)}
          className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-200
            ${isDark
              ? 'hover:bg-white/[0.08] text-white/90 hover:text-white'
              : 'hover:bg-black/[0.05] text-black/80 hover:text-black'}`}
        >
          {/* Status indicator with subtle glow */}
          <span className="relative flex items-center justify-center">
            {/* Subtle glow effect */}
            <span
              className={`absolute w-3 h-3 rounded-full blur-[3px] opacity-40 ${statusInfo.color} ${statusInfo.animate ? 'animate-pulse' : ''}`}
            />
            {/* Core dot */}
            <span
              className={`relative w-1.5 h-1.5 rounded-full ${statusInfo.color} ${statusInfo.animate ? 'animate-pulse' : ''}`}
              title={statusInfo.label}
            />
          </span>
          <span className="text-[15px] font-medium">ShikshaSetu</span>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium tracking-wider uppercase
            ${policyDisplay.bg} ${policyDisplay.color}`}>
            {policyDisplay.label}
          </span>
          <ChevronDown className={`w-4 h-4 ml-0.5 opacity-50`} />
        </button>

        {/* Right side - Theme toggle only */}
        <div className="flex items-center">
          <button
            onClick={() => setTheme(isDark ? 'light' : 'dark')}
            className={`p-2 rounded-full transition-all duration-200
              ${isDark
                ? 'hover:bg-white/[0.08] active:bg-white/[0.12] text-white/60 hover:text-white'
                : 'hover:bg-black/[0.05] active:bg-black/[0.08] text-black/50 hover:text-black'}`}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </div>
      </div>
    </header>
  );
}
