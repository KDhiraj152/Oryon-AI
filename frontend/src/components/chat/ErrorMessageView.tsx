/**
 * ErrorMessageView — Error-state display for failed messages.
 */
import { AlertCircle, RefreshCw } from 'lucide-react';
import type { Message } from '../../store/chatStore';
import { themed } from './chatHelpers';

export function ErrorMessageView({ message, isDark, onRetry }: Readonly<{
  message: Message; isDark: boolean; onRetry?: (messageId: string) => void;
}>) {
  return (
    <div className="w-full animate-message-in py-4">
      <div className="w-full max-w-3xl mx-auto px-4">
        <div className="flex gap-4">
          <div className={`w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center
            ${themed(isDark, 'bg-red-500/20', 'bg-red-100')}`}>
            <AlertCircle className={`w-4 h-4 ${themed(isDark, 'text-red-400', 'text-red-500')}`} />
          </div>
          <div className="flex-1 min-w-0 space-y-3 overflow-hidden pt-1">
            <div className={`text-[15px] leading-relaxed break-words ${themed(isDark, 'text-red-300', 'text-red-600')}`}>
              {message.content}
            </div>
            {onRetry && (
              <button onClick={() => onRetry(message.id)}
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-200
                  ${themed(isDark, 'bg-white/[0.08] hover:bg-white/[0.12] text-white/80', 'bg-gray-100 hover:bg-gray-200 text-gray-700')}`}>
                <RefreshCw className="w-4 h-4" aria-hidden="true" />
                Retry
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
