import { memo, useCallback } from 'react';
import { OmLogo } from '../landing/OmLogo';

interface EmptyStateProps {
  readonly isDark: boolean;
  readonly onQuickAction: (prompt: string) => void;
}

// Static capability cards - defined outside component for stability
const capabilities = [
  {
    label: 'Explain anything',
    description: 'Explain how machine learning works in sim...',
    prompt: 'Explain how machine learning works in simple terms with examples',
  },
  {
    label: 'Help me write',
    description: 'Help me write a professional email to reque...',
    prompt: 'Help me write a professional email to request a meeting',
  },
  {
    label: 'Code with me',
    description: 'Write a Python function that finds all prime ...',
    prompt: 'Write a Python function that finds all prime numbers up to n',
  },
  {
    label: 'Translate',
    description: 'Translate to Hindi: The future belongs to th...',
    prompt: 'Translate to Hindi: The future belongs to those who believe in the beauty of their dreams.',
  },
] as const;

// Memoized capability button
const CapabilityButton = memo(function CapabilityButton({
  label,
  description,
  isDark,
  onClick,
}: {
  label: string;
  description: string;
  isDark: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`group p-5 rounded-2xl text-left transition-all duration-200
        focus-visible:outline-none focus-visible:ring-2
        ${isDark
          ? 'bg-white/[0.03] hover:bg-white/[0.05] border border-white/[0.06] hover:border-white/[0.10] focus-visible:ring-white/20'
          : 'bg-white hover:bg-gray-50 border border-gray-200/80 hover:border-gray-300 focus-visible:ring-black/10'
        }`}
    >
      <span className={`block text-[15px] font-medium mb-1
        ${isDark ? 'text-white/90' : 'text-gray-900'}`}>
        {label}
      </span>
      <span className={`block text-[13px] leading-relaxed
        ${isDark ? 'text-white/35' : 'text-gray-500'}`}>
        {description}
      </span>
    </button>
  );
});

export const EmptyState = memo(function EmptyState({ isDark, onQuickAction }: EmptyStateProps) {
  // Memoize click handlers to prevent recreation on each render
  const handleClick = useCallback((prompt: string) => () => onQuickAction(prompt), [onQuickAction]);

  return (
    <div className="h-full flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-3xl mx-auto">
        {/* Centered Logo with subtle glow */}
        <div className="flex justify-center mb-10">
          <div className="relative group">
            <div className={`absolute inset-0 rounded-full blur-3xl transition-opacity duration-500
              ${isDark ? 'bg-emerald-500/10 group-hover:bg-emerald-500/15' : 'bg-emerald-200/20 group-hover:bg-emerald-200/30'}`} />
            <div className={`relative inline-flex items-center justify-center w-14 h-14 rounded-2xl
              ${isDark ? 'bg-white/[0.03]' : 'bg-gray-100/80'}`}>
              <OmLogo variant="minimal" size={28} color={isDark ? 'dark' : 'light'} animated />
            </div>
          </div>
        </div>

        {/* Large, minimal greeting - confident statement */}
        <h1 className={`text-4xl sm:text-5xl font-semibold text-center mb-10 tracking-tight
          ${isDark ? 'text-white' : 'text-gray-900'}`}>
          What can I help you with?
        </h1>

        {/* Capability Cards - 2x2 grid, subtle glass morphism */}
        <div className="grid grid-cols-2 gap-3 max-w-2xl mx-auto">
          {capabilities.map((item) => (
            <CapabilityButton
              key={item.label}
              label={item.label}
              description={item.description}
              isDark={isDark}
              onClick={handleClick(item.prompt)}
            />
          ))}
        </div>

        {/* Subtle capabilities footer */}
        <div className={`flex items-center justify-center gap-6 mt-12 text-[11px] font-medium tracking-widest uppercase
          ${isDark ? 'text-white/15' : 'text-black/20'}`}>
          <span className="flex items-center gap-2">
            <span className="w-1 h-1 rounded-full bg-current" />
            Code
          </span>
          <span className="flex items-center gap-2">
            <span className="w-1 h-1 rounded-full bg-current" />
            Voice
          </span>
          <span className="flex items-center gap-2">
            <span className="w-1 h-1 rounded-full bg-current" />
            Documents
          </span>
          <span className="flex items-center gap-2">
            <span className="w-1 h-1 rounded-full bg-current" />
            22 Languages
          </span>
        </div>
      </div>
    </div>
  );
});
