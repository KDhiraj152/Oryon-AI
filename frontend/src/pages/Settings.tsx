import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Volume2,
  Moon,
  Sun,
  Monitor,
  Palette,
  Shield,
  Trash2,
  LogOut,
  User,
  UserCircle,
  Sparkles,
  BookOpen,
  FlaskConical,
  Lock
} from 'lucide-react';
import { useThemeStore, useAuthStore, useChatStore, useProfileStore } from '../store';
import { OmLogo } from '../components/landing/OmLogo';
import { useSystemStatus } from '../context/SystemStatusContext';

const VOICE_TYPES = [
  { value: 'female', label: 'Female', Icon: User },
  { value: 'male', label: 'Male', Icon: UserCircle },
];

const SPEECH_SPEEDS = [
  { value: 0.75, label: '0.75×', desc: 'Slow' },
  { value: 1, label: '1×', desc: 'Normal' },
  { value: 1.25, label: '1.25×', desc: 'Fast' },
  { value: 1.5, label: '1.5×', desc: 'Faster' },
];

// Settings store (persisted)
interface UserSettings {
  voiceType: string;
  speechSpeed: number;
  autoReadResponses: boolean;
}

const DEFAULT_SETTINGS: UserSettings = {
  voiceType: 'female',
  speechSpeed: 1,
  autoReadResponses: false,
};

function loadSettings(): UserSettings {
  try {
    const saved = localStorage.getItem('user-settings');
    return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function saveSettings(settings: UserSettings) {
  localStorage.setItem('user-settings', JSON.stringify(settings));
}

// Style constants to avoid conditional branches
const DARK_FOCUS = 'focus-visible:ring-white focus-visible:ring-offset-[#0a0a0a]';
const LIGHT_FOCUS = 'focus-visible:ring-gray-400 focus-visible:ring-offset-gray-50';

const FOCUS_STYLES = { dark: DARK_FOCUS, light: LIGHT_FOCUS } as const;

function getFocusStyle(isDark: boolean): string {
  return FOCUS_STYLES[isDark ? 'dark' : 'light'];
}

// Delete Confirmation Modal Component
interface DeleteModalProps {
  readonly isOpen: boolean;
  readonly onClose: () => void;
  readonly onConfirm: () => void;
  readonly conversationCount: number;
  readonly isDark: boolean;
}

function DeleteConfirmModal({ isOpen, onClose, onConfirm, conversationCount, isDark }: DeleteModalProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
    } else {
      if (dialog.open) {
        dialog.close();
      }
    }
  }, [isOpen]);

  const handleClose = useCallback(() => {
    onClose();
  }, [onClose]);

  const pluralSuffix = conversationCount === 1 ? '' : 's';

  return (
    <dialog
      ref={dialogRef}
      onClose={handleClose}
      className="fixed inset-0 z-modal flex items-center justify-center p-4 bg-transparent backdrop:bg-black/60 backdrop:backdrop-blur-md m-auto open:flex"
    >
      <div
        className={`w-full max-w-md rounded-modal p-6 shadow-2xl animate-scaleIn
          ${isDark ? 'bg-[#0a0a0a] border border-white/10' : 'bg-white'}`}
      >
        <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-4 ${isDark ? 'bg-red-500/10' : 'bg-red-50'}`}>
          <Trash2 className={`w-5 h-5 ${isDark ? 'text-red-400' : 'text-red-600'}`} aria-hidden="true" />
        </div>

        <h3 id="delete-modal-title" className={`text-title font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Clear All Chat History?
        </h3>
        <p className={`text-body-sm mb-6 ${isDark ? 'text-white/60' : 'text-gray-600'}`}>
          This will permanently delete all {conversationCount} conversation{pluralSuffix}. This action cannot be undone.
        </p>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            autoFocus
            className={`flex-1 px-4 min-h-touch py-2.5 rounded-btn text-body-sm font-medium
              transition-all duration-fast active:scale-[0.98]
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
              ${isDark
                ? 'bg-white/10 text-white hover:bg-white/15 focus-visible:ring-white focus-visible:ring-offset-[#0a0a0a]'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200 focus-visible:ring-gray-400 focus-visible:ring-offset-white'}`}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 px-4 min-h-touch py-2.5 rounded-btn text-body-sm font-medium
              bg-red-500 text-white hover:bg-red-600
              transition-all duration-fast active:scale-[0.98]
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0a0a]"
          >
            Delete All
          </button>
        </div>
      </div>
    </dialog>
  );
}

// Policy Mode Types
type PolicyMode = 'OPEN' | 'EDUCATION' | 'RESEARCH' | 'RESTRICTED';

const POLICY_MODES: {
  value: PolicyMode;
  label: string;
  desc: string;
  icon: typeof Sparkles;
  color: string;
}[] = [
  {
    value: 'OPEN',
    label: 'Open',
    desc: 'General AI with essential safety',
    icon: Sparkles,
    color: 'emerald'
  },
  {
    value: 'EDUCATION',
    label: 'Education',
    desc: 'NCERT curriculum aligned',
    icon: BookOpen,
    color: 'blue'
  },
  {
    value: 'RESEARCH',
    label: 'Research',
    desc: 'Maximum freedom for academics',
    icon: FlaskConical,
    color: 'purple'
  },
  {
    value: 'RESTRICTED',
    label: 'Restricted',
    desc: 'Full policy enforcement',
    icon: Lock,
    color: 'amber'
  },
];

// Policy Mode Section Component
function PolicyModeSection({ isDark }: { isDark: boolean }) {
  const { policy, switchPolicyMode, isSwitchingPolicy } = useSystemStatus();
  const [error, setError] = useState<string | null>(null);

  const currentMode = (policy?.mode || 'OPEN') as PolicyMode;

  const handleModeSwitch = async (mode: PolicyMode) => {
    if (mode === currentMode || isSwitchingPolicy) return;

    setError(null);
    try {
      await switchPolicyMode(mode);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to switch mode');
    }
  };

  return (
    <section aria-labelledby="policy-heading">
      <div className="flex items-center gap-2 mb-4 px-1">
        <Sparkles className={`w-4 h-4 ${isDark ? 'text-white/60' : 'text-gray-600'}`} aria-hidden="true" />
        <h2 id="policy-heading" className={`text-sm font-semibold uppercase tracking-wider ${isDark ? 'text-white/60' : 'text-gray-500'}`}>
          AI Mode
        </h2>
      </div>

      <div className={`p-6 rounded-3xl border ${isDark ? 'bg-white/[0.03] border-white/[0.05]' : 'bg-white border-gray-100 shadow-sm'}`}>
        <div className="mb-4">
          <span className={`block text-base font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Operating Mode
          </span>
          <p className={`text-sm mt-1 ${isDark ? 'text-white/40' : 'text-gray-500'}`}>
            {policy?.description || 'Safe AI without restrictions'}
          </p>
        </div>

        {error && (
          <div className={`mb-4 p-3 rounded-xl text-sm ${isDark ? 'bg-red-500/10 text-red-400' : 'bg-red-50 text-red-600'}`}>
            {error}
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          {POLICY_MODES.map((mode) => {
            const isSelected = currentMode === mode.value;
            const IconComponent = mode.icon;

            const colorStyles = {
              emerald: isDark ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400' : 'bg-emerald-50 border-emerald-200 text-emerald-700',
              blue: isDark ? 'bg-blue-500/20 border-blue-500/30 text-blue-400' : 'bg-blue-50 border-blue-200 text-blue-700',
              purple: isDark ? 'bg-purple-500/20 border-purple-500/30 text-purple-400' : 'bg-purple-50 border-purple-200 text-purple-700',
              amber: isDark ? 'bg-amber-500/20 border-amber-500/30 text-amber-400' : 'bg-amber-50 border-amber-200 text-amber-700',
            };

            return (
              <button
                key={mode.value}
                onClick={() => handleModeSwitch(mode.value)}
                disabled={isSwitchingPolicy}
                aria-pressed={isSelected ? "true" : "false"}
                className={`relative flex flex-col items-center justify-center py-4 px-3 rounded-2xl text-sm font-medium border
                  transition-all duration-200 disabled:opacity-50
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
                  ${isDark
                    ? 'focus-visible:ring-white focus-visible:ring-offset-[#0a0a0a]'
                    : 'focus-visible:ring-gray-400 focus-visible:ring-offset-white'}
                  ${isSelected
                    ? colorStyles[mode.color as keyof typeof colorStyles]
                    : (isDark ? 'bg-white/5 border-white/10 text-white/60 hover:bg-white/10' : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100')}`}
              >
                {isSwitchingPolicy && isSelected && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/20 rounded-2xl">
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  </div>
                )}
                <IconComponent className="w-5 h-5 mb-2" aria-hidden="true" />
                <span className="font-semibold">{mode.label}</span>
                <span className={`text-[11px] mt-1 text-center leading-tight ${isSelected ? 'opacity-80' : 'opacity-50'}`}>
                  {mode.desc}
                </span>
              </button>
            );
          })}
        </div>

        {/* Safety notice */}
        <div className={`mt-4 pt-4 border-t ${isDark ? 'border-white/5' : 'border-gray-100'}`}>
          <p className={`text-xs flex items-center gap-1.5 ${isDark ? 'text-white/30' : 'text-gray-400'}`}>
            <Shield className="w-3.5 h-3.5" />
            All modes maintain safety: blocking violence, weapons & exploitation content
          </p>
        </div>
      </div>
    </section>
  );
}

export default function Settings() {
  const navigate = useNavigate();
  const { resolvedTheme, theme, setTheme } = useThemeStore();
  const { user, logout } = useAuthStore();
  const { conversations } = useChatStore();
  const { fetchProfile } = useProfileStore();
  const isDark = resolvedTheme === 'dark';

  const [settings, setSettings] = useState<UserSettings>(loadSettings);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  // Load profile on mount
  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  // Save settings on change
  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  const updateSetting = <K extends keyof UserSettings>(key: K, value: UserSettings[K]) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleClearHistory = () => {
    localStorage.removeItem('chat-storage');
    setShowDeleteConfirm(false);
    navigate('/chat');
  };

  const handleLogout = () => {
    logout();
    navigate('/landing');
  };

  return (
    <div className={`h-full overflow-y-auto ${isDark
      ? 'bg-[#0a0a0a] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900/40 via-[#0a0a0a] to-[#0a0a0a]'
      : 'bg-white bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-50 via-white to-white'}`}
    >
      {/* Skip link for accessibility */}
      <a href="#settings-content" className="skip-link">Skip to settings</a>

      {/* Header - Minimal like ChatGPT */}
      <header className={`sticky top-0 z-sticky border-b ${isDark ? 'border-white/5 bg-black/80' : 'border-gray-200/50 bg-white/80'} backdrop-blur-2xl`}>
        <div className="max-w-3xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className={`p-2.5 -ml-2 rounded-full flex items-center justify-center
              transition-colors duration-200
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
              ${isDark
                ? 'hover:bg-white/10 text-white/70 hover:text-white focus-visible:ring-white focus-visible:ring-offset-black'
                : 'hover:bg-gray-100 text-gray-600 hover:text-gray-900 focus-visible:ring-gray-400 focus-visible:ring-offset-white'}`}
            aria-label="Go back"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className={`text-lg font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>Settings</h1>
        </div>
      </header>

      <main id="settings-content" className="max-w-2xl mx-auto px-4 sm:px-6 py-8 space-y-8">
        {/* Profile Section - Clean Card */}
        <section
          className={`p-6 rounded-3xl border ${isDark ? 'bg-white/[0.03] border-white/[0.05]' : 'bg-white border-gray-100 shadow-sm'}`}
          aria-label="Profile information"
        >
          <div className="flex items-center gap-5">
            <div className={`w-14 h-14 rounded-full flex items-center justify-center text-xl font-semibold shrink-0
              ${isDark ? 'bg-white/10 text-white' : 'bg-gray-900 text-white'}`}
              aria-hidden="true"
            >
              {user?.name?.charAt(0).toUpperCase() || 'U'}
            </div>
            <div className="flex-1 min-w-0">
              <div className={`text-lg font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                {user?.name || 'Guest User'}
              </div>
              <div className={`text-sm ${isDark ? 'text-white/40' : 'text-gray-500'}`}>
                {user?.email || 'Not signed in'}
              </div>
            </div>
            <OmLogo variant="minimal" size={32} color={isDark ? 'dark' : 'light'} animated={false} />
          </div>
        </section>

        {/* Appearance - Premium Toggle */}
        <section aria-labelledby="appearance-heading">
          <div className="flex items-center gap-2 mb-4 px-1">
            <Palette className={`w-4 h-4 ${isDark ? 'text-white/60' : 'text-gray-600'}`} aria-hidden="true" />
            <h2 id="appearance-heading" className={`text-sm font-semibold uppercase tracking-wider ${isDark ? 'text-white/60' : 'text-gray-500'}`}>
              Appearance
            </h2>
          </div>

          <div
            className={`p-1.5 rounded-full border inline-flex ${isDark ? 'bg-black/40 border-white/10' : 'bg-gray-100/50 border-gray-200'}`}
            role="radiogroup"
            aria-label="Theme selection"
          >
            {[
              { value: 'light', label: 'Light', icon: Sun },
              { value: 'dark', label: 'Dark', icon: Moon },
              { value: 'system', label: 'System', icon: Monitor },
            ].map((option) => {
              const isActive = theme === option.value;
              const activeStyle = isDark ? 'bg-white/10 text-white shadow-sm' : 'bg-white text-gray-900 shadow-sm';
              const inactiveStyle = isDark ? 'text-white/50 hover:text-white/70' : 'text-gray-500 hover:text-gray-900';
              return (
                <button
                  key={option.value}
                  onClick={() => setTheme(option.value as 'light' | 'dark' | 'system')}
                  aria-pressed={isActive ? "true" : "false"}
                  className={`flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium
                    transition-all duration-200
                    focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
                    ${isDark
                      ? 'focus-visible:ring-white focus-visible:ring-offset-black'
                      : 'focus-visible:ring-gray-400 focus-visible:ring-offset-gray-100'}
                    ${isActive ? activeStyle : inactiveStyle}`}
                >
                  <option.icon className="w-4 h-4" aria-hidden="true" />
                  {option.label}
                </button>
              );
            })}
          </div>
        </section>

        {/* Voice & Audio - Modern Design */}
        <section aria-labelledby="voice-heading">
          <div className="flex items-center gap-2 mb-4 px-1">
            <Volume2 className={`w-4 h-4 ${isDark ? 'text-white/60' : 'text-gray-600'}`} aria-hidden="true" />
            <h2 id="voice-heading" className={`text-sm font-semibold uppercase tracking-wider ${isDark ? 'text-white/60' : 'text-gray-500'}`}>
              Voice & Audio
            </h2>
          </div>

          <div className="space-y-4">
            {/* Voice Type */}
            <div className={`p-6 rounded-3xl border ${isDark ? 'bg-white/[0.03] border-white/[0.05]' : 'bg-white border-gray-100 shadow-sm'}`}>
              <span className={`block text-base font-medium mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`} id="voice-type-label">
                Voice Type
              </span>
              <div className="grid grid-cols-2 gap-3" aria-labelledby="voice-type-label">
                {VOICE_TYPES.map((voice) => {
                  const isSelected = settings.voiceType === voice.value;
                  return (
                    <button
                      key={voice.value}
                      onClick={() => updateSetting('voiceType', voice.value)}
                      aria-pressed={isSelected ? "true" : "false"}
                      className={`flex items-center justify-center gap-2 py-3 rounded-2xl text-sm font-medium
                        transition-all duration-200
                        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
                        ${getFocusStyle(isDark)}
                        ${isSelected
                          ? (isDark ? 'bg-white text-black' : 'bg-gray-900 text-white')
                          : (isDark ? 'bg-white/5 text-white/60 hover:bg-white/10' : 'bg-gray-50 text-gray-600 hover:bg-gray-100')}`}
                    >
                      <voice.Icon className="w-4 h-4" aria-hidden="true" />
                      {voice.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Speech Speed */}
            <div className={`p-6 rounded-3xl border ${isDark ? 'bg-white/[0.03] border-white/[0.05]' : 'bg-white border-gray-100 shadow-sm'}`}>
              <span className={`block text-base font-medium mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`} id="speed-label">
                Speech Speed
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3" aria-labelledby="speed-label">
                {SPEECH_SPEEDS.map((speed) => {
                  const isSelected = settings.speechSpeed === speed.value;
                  return (
                    <button
                      key={speed.value}
                      onClick={() => updateSetting('speechSpeed', speed.value)}
                      aria-pressed={isSelected ? "true" : "false"}
                      className={`py-3 rounded-2xl text-sm font-medium
                        transition-all duration-200
                        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
                        ${getFocusStyle(isDark)}
                        ${isSelected
                          ? (isDark ? 'bg-white text-black' : 'bg-gray-900 text-white')
                          : (isDark ? 'bg-white/5 text-white/60 hover:bg-white/10' : 'bg-gray-50 text-gray-600 hover:bg-gray-100')}`}
                    >
                      <div>{speed.label}</div>
                      <div className={`text-[10px] font-normal mt-0.5 ${isSelected ? 'opacity-80' : 'opacity-50'}`}>
                        {speed.desc}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Auto Read Toggle */}
            <div className={`p-6 rounded-3xl border ${isDark ? 'bg-white/[0.03] border-white/[0.05]' : 'bg-white border-gray-100 shadow-sm'}`}>
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <label
                    htmlFor="auto-read-toggle"
                    className={`text-base font-medium block cursor-pointer ${isDark ? 'text-white' : 'text-gray-900'}`}
                  >
                    Auto-read responses
                  </label>
                  <p className={`text-sm mt-1 ${isDark ? 'text-white/40' : 'text-gray-500'}`}>
                    Automatically play audio for AI responses
                  </p>
                </div>
                {(() => {
                  const focusStyle = isDark
                    ? 'focus-visible:ring-white focus-visible:ring-offset-[#0a0a0a]'
                    : 'focus-visible:ring-gray-400 focus-visible:ring-offset-gray-50';
                  const trackActiveStyle = isDark ? 'bg-white' : 'bg-gray-900';
                  const trackInactiveStyle = isDark ? 'bg-white/10' : 'bg-gray-200';
                  const thumbActiveStyle = isDark ? 'bg-black' : 'bg-white';

                  return (
                    <button
                      id="auto-read-toggle"
                      type="button"
                      role="switch"
                      aria-checked={settings.autoReadResponses ? "true" : "false"}
                      onClick={() => updateSetting('autoReadResponses', !settings.autoReadResponses)}
                      className={`relative w-14 h-8 rounded-full transition-all duration-200 shrink-0 ml-4
                        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
                        ${focusStyle}
                        ${settings.autoReadResponses ? trackActiveStyle : trackInactiveStyle}`}
                    >
                      <span
                        className={`absolute top-1 left-1 w-6 h-6 rounded-full transition-transform duration-200 shadow-sm
                          ${settings.autoReadResponses
                            ? `translate-x-6 ${thumbActiveStyle}`
                            : 'translate-x-0 bg-white'}`}
                        aria-hidden="true"
                      />
                    </button>
                  );
                })()}
              </div>
            </div>
          </div>
        </section>

        {/* AI Policy Mode - New Section */}
        <PolicyModeSection isDark={isDark} />

        {/* Data & Privacy - Danger Zone */}
        <section aria-labelledby="privacy-heading">
          <div className="flex items-center gap-2 mb-4 px-1">
            <Shield className={`w-4 h-4 ${isDark ? 'text-white/60' : 'text-gray-600'}`} aria-hidden="true" />
            <h2 id="privacy-heading" className={`text-sm font-semibold uppercase tracking-wider ${isDark ? 'text-white/60' : 'text-gray-500'}`}>
              Data & Privacy
            </h2>
          </div>

          <div className="space-y-4">
            {/* Chat History Info */}
            <div className={`p-6 rounded-3xl border ${isDark ? 'bg-white/[0.03] border-white/[0.05]' : 'bg-white border-gray-100 shadow-sm'}`}>
              <div className={`text-base font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Chat History
              </div>
              <p className={`text-sm mt-1 ${isDark ? 'text-white/40' : 'text-gray-500'}`}>
                {conversations.length} conversation{conversations.length === 1 ? '' : 's'} stored locally on your device
              </p>
            </div>

            {/* Clear History - Danger Button */}
            <div className={`p-2 rounded-3xl border ${isDark ? 'bg-red-500/5 border-red-500/10' : 'bg-red-50 border-red-100'}`}>
              <button
                onClick={() => setShowDeleteConfirm(true)}
                className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-2xl text-sm font-medium
                  transition-all duration-200 active:scale-[0.98]
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2
                  ${isDark
                    ? 'text-red-400 hover:bg-red-500/10 focus-visible:ring-offset-[#0a0a0a]'
                    : 'text-red-700 hover:bg-red-100 focus-visible:ring-offset-red-50'
                  }`}
              >
                <Trash2 className="w-4 h-4" aria-hidden="true" />
                Clear All Chat History
              </button>
            </div>
          </div>
        </section>

        {/* Account - Minimal */}
        <section aria-label="Account actions">
          <button
            onClick={handleLogout}
            className={`w-full flex items-center justify-between p-6 rounded-3xl border
              transition-all duration-200 active:scale-[0.99]
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
              ${isDark
                ? 'bg-white/[0.03] border-white/[0.05] hover:bg-white/[0.05] focus-visible:ring-white focus-visible:ring-offset-black'
                : 'bg-white border-gray-100 hover:bg-gray-50 focus-visible:ring-gray-400 focus-visible:ring-offset-white shadow-sm'}`}
          >
            <div className="flex items-center gap-3">
              <LogOut className={`w-5 h-5 ${isDark ? 'text-white/60' : 'text-gray-600'}`} aria-hidden="true" />
              <span className={`text-base font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>Sign Out</span>
            </div>
          </button>
        </section>

        {/* Footer - Subtle */}
        <footer className={`text-center py-8 ${isDark ? 'text-white/20' : 'text-gray-400'}`}>
          <p className="text-xs">Shiksha Setu v1.0.0</p>
          <p className="text-xs mt-1">Made with ❤️ for Indian Education</p>
        </footer>
      </main>

      {/* Delete Confirmation Modal - Only render when needed */}
      {showDeleteConfirm && (
        <DeleteConfirmModal
          isOpen={showDeleteConfirm}
          onClose={() => setShowDeleteConfirm(false)}
          onConfirm={handleClearHistory}
          conversationCount={conversations.length}
          isDark={isDark}
        />
      )}
    </div>
  );
}
