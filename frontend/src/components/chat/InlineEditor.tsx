/**
 * InlineEditor — Edit previous user messages in-place.
 *
 * UX flow:
 * 1. User clicks edit on a user message
 * 2. Message content transforms into an editable textarea
 * 3. User modifies and presses Enter (or clicks Save)
 * 4. Creates a new branch — original message preserved
 * 5. Assistant re-generates response for the edited message
 *
 * This creates conversation branching — a core UX differentiator.
 */

import { memo, useState, useCallback, useRef, useEffect } from 'react';
import { Check, X } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';

interface InlineEditorProps {
  /** Current message content */
  readonly content: string;
  /** Called when user confirms edit */
  readonly onSave: (newContent: string) => void;
  /** Called when user cancels edit */
  readonly onCancel: () => void;
}

export const InlineEditor = memo(function InlineEditor({
  content,
  onSave,
  onCancel,
}: InlineEditorProps) {
  const [value, setValue] = useState(content);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  // Auto-focus and select all on mount
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.focus();
      el.setSelectionRange(el.value.length, el.value.length);
      // Auto-resize
      el.style.height = 'auto';
      el.style.height = `${el.scrollHeight}px`;
    }
  }, []);

  // Auto-resize on content change
  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }, []);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && value.trim() !== content.trim()) {
        onSave(value.trim());
      } else {
        onCancel();
      }
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      onCancel();
    }
  }, [value, content, onSave, onCancel]);

  const handleSave = useCallback(() => {
    if (value.trim() && value.trim() !== content.trim()) {
      onSave(value.trim());
    } else {
      onCancel();
    }
  }, [value, content, onSave, onCancel]);

  const hasChanges = value.trim() !== content.trim();

  return (
    <div className="animate-scale-up">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        rows={1}
        className={`w-full resize-none rounded-xl px-4 py-3 text-[15px] leading-relaxed border-2 outline-none transition-all duration-200
          ${isDark
            ? 'bg-white/[0.04] border-white/[0.12] text-white/90 focus:border-white/25 placeholder:text-white/20'
            : 'bg-white border-gray-200 text-gray-800 focus:border-gray-400 placeholder:text-gray-400 shadow-sm'}`}
        aria-label="Edit message"
      />

      {/* Save / Cancel bar */}
      <div className="flex items-center justify-end gap-2 mt-2">
        <span className={`text-[11px] mr-auto ${isDark ? 'text-white/25' : 'text-gray-400'}`}>
          Enter to save · Esc to cancel
        </span>
        <button
          onClick={onCancel}
          className={`p-2 rounded-lg transition-colors ${isDark ? 'text-white/40 hover:text-white/60 hover:bg-white/[0.06]' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'}`}
          aria-label="Cancel editing"
          title="Cancel"
        >
          <X className="w-4 h-4" />
        </button>
        <button
          onClick={handleSave}
          disabled={!hasChanges}
          className={`p-2 rounded-lg transition-colors
            ${hasChanges
              ? isDark
                ? 'text-white/80 hover:text-white hover:bg-white/[0.08]'
                : 'text-gray-700 hover:text-gray-900 hover:bg-gray-100'
              : isDark
                ? 'text-white/15 cursor-not-allowed'
                : 'text-gray-300 cursor-not-allowed'
            }`}
          aria-label="Save edit"
          title="Save"
        >
          <Check className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
});
