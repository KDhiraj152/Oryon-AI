/**
 * MessageAvatar — User/assistant avatar with streaming animation.
 */
import { User } from 'lucide-react';
import { OmLogo } from '../landing/OmLogo';
import { themed } from './chatHelpers';

export function MessageAvatar({ isUser, isDark, isStreaming }: Readonly<{
  isUser: boolean; isDark: boolean; isStreaming: boolean;
}>) {
  return (
    <div className={`w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center transition-all duration-200
      ${themed(isDark, isUser ? 'bg-white/[0.08]' : 'bg-white/[0.06]', 'bg-gray-100')}
      ${isStreaming ? 'animate-avatar-pop' : ''}`}>
      {isUser
        ? <User className={`w-4 h-4 ${themed(isDark, 'text-white/60', 'text-gray-500')}`} />
        : <OmLogo variant="minimal" size={16} color={isDark ? 'dark' : 'light'} animated={isStreaming} />
      }
    </div>
  );
}
