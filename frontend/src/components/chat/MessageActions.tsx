/**
 * MessageActions — Action buttons toolbar for chat messages.
 */
import { Copy, RefreshCw, Volume2, VolumeX, Check, Pencil, Loader2 } from 'lucide-react';
import { themed, getAudioAriaLabel, getAudioTitle } from './chatHelpers';
import { FeedbackButtons } from './FeedbackButtons';
import type { Message } from '../../store/chatStore';

export function MessageActions({ isDark, copied, onCopy, onRetry, onAudio, onStartEdit, isPlayingAudio, isLoadingAudio, messageId, feedback }: Readonly<{
  isDark: boolean; copied: boolean;
  onCopy?: () => void; onRetry?: () => void;
  onAudio?: (e: React.MouseEvent) => void; onStartEdit?: () => void;
  isPlayingAudio: boolean; isLoadingAudio: boolean;
  messageId: string; feedback?: Message['feedback'];
}>) {
  const inactive = themed(isDark,
    'text-white/30 hover:text-white/70 hover:bg-white/[0.08]',
    'text-gray-400 hover:text-gray-600 hover:bg-gray-100');

  return (
    <div className="flex items-center gap-1 pt-3 opacity-0 group-hover:opacity-100 transition-all duration-200"
      role="toolbar" aria-label="Message actions">
      {onCopy && (
        <button onClick={onCopy}
          className={`p-2 rounded-full transition-all duration-200 ${copied ? 'text-emerald-500 bg-emerald-500/10' : inactive}`}
          aria-label={copied ? 'Copied' : 'Copy'} title="Copy">
          {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
        </button>
      )}
      {onStartEdit && (
        <button onClick={onStartEdit}
          className={`p-2 rounded-full transition-all duration-200 ${inactive}`}
          aria-label="Edit message" title="Edit">
          <Pencil className="w-4 h-4" />
        </button>
      )}
      {onRetry && (
        <button onClick={onRetry}
          className={`p-2 rounded-full transition-all duration-200 ${inactive}`}
          aria-label="Regenerate" title="Regenerate">
          <RefreshCw className="w-4 h-4" />
        </button>
      )}
      {onAudio && (
        <button onClick={onAudio}
          className={`p-2 rounded-full transition-all duration-200 ${
            isPlayingAudio || isLoadingAudio
              ? themed(isDark, 'text-orange-400 bg-orange-500/15', 'text-orange-500 bg-orange-50')
              : inactive
          }`}
          aria-label={getAudioAriaLabel(isPlayingAudio, isLoadingAudio)}
          title={getAudioTitle(isPlayingAudio, isLoadingAudio)}>
          {isLoadingAudio && <Loader2 className="w-4 h-4 animate-spin" />}
          {!isLoadingAudio && isPlayingAudio && <VolumeX className="w-4 h-4" />}
          {!isLoadingAudio && !isPlayingAudio && <Volume2 className="w-4 h-4" />}
        </button>
      )}
      <FeedbackButtons messageId={messageId} feedback={feedback} />
    </div>
  );
}
