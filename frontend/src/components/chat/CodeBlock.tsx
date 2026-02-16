/**
 * CodeBlock — Syntax-highlighted code block with copy button.
 */
import { useState, useCallback, memo } from 'react';
import { Copy, Check } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';
import { themed } from './chatHelpers';
import { LANG_DISPLAY } from './codeLanguages';

interface CodeBlockProps {
  readonly language?: string;
  readonly children: string;
}

export const CodeBlock = memo(function CodeBlock({ language, children }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [children]);

  return (
    <div className={`rounded-xl overflow-hidden my-5 border font-mono text-[13px] leading-relaxed shadow-sm
      ${themed(isDark, 'bg-[#1e1e1e] border-white/10', 'bg-white border-gray-200')}`}>
      {/* Header bar */}
      <div className={`flex items-center justify-between px-4 py-2.5 border-b
        ${themed(isDark, 'bg-[#252525] border-white/5', 'bg-gray-50 border-gray-100')}`}>
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <div className={`w-2.5 h-2.5 rounded-full ${themed(isDark, 'bg-white/20', 'bg-gray-300')}`} />
            <div className={`w-2.5 h-2.5 rounded-full ${themed(isDark, 'bg-white/20', 'bg-gray-300')}`} />
            <div className={`w-2.5 h-2.5 rounded-full ${themed(isDark, 'bg-white/20', 'bg-gray-300')}`} />
          </div>
          <span className={`text-xs font-medium tracking-wide ${themed(isDark, 'text-gray-400', 'text-gray-500')}`}>
            {LANG_DISPLAY[language?.toLowerCase() || ''] || language || 'Code'}
          </span>
        </div>
        <button
          onClick={handleCopy}
          className={`flex items-center gap-1.5 text-xs font-medium transition-all duration-200 px-2.5 py-1.5 rounded-lg
            ${copied
              ? 'text-emerald-500 bg-emerald-500/10'
              : themed(isDark,
                  'text-gray-400 hover:text-white hover:bg-white/10',
                  'text-gray-500 hover:text-gray-900 hover:bg-gray-200')}`}
          aria-label={copied ? 'Copied!' : 'Copy code'}
        >
          {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      {/* Code content */}
      <div className="overflow-x-auto p-5">
        <pre className={`m-0 bg-transparent font-mono text-[13px] leading-relaxed
          ${themed(isDark, 'text-white', 'text-gray-800')}`}>
          {children}
        </pre>
      </div>
    </div>
  );
});
