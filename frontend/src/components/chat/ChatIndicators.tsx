import { RefreshCw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { OmLogo } from '../landing/OmLogo';

interface ThinkingIndicatorProps {
  readonly isDark: boolean;
}

interface RegeneratingIndicatorProps {
  readonly isDark: boolean;
  readonly streamingMessage: string;
}

/**
 * Shimmer loading indicator shown while AI is processing
 */
export function ThinkingIndicator({ isDark }: ThinkingIndicatorProps) {
  return (
    <output className="py-6 block" aria-label="AI is thinking">
      <div className="w-full max-w-3xl mx-auto px-4">
        <div className="flex gap-4">
          <div className={`w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center
            ${isDark ? 'bg-white/[0.06]' : 'bg-gray-100'}`}>
            <OmLogo variant="minimal" size={16} color={isDark ? 'dark' : 'light'} animated={true} />
          </div>
          <div className="flex-1 pt-0.5">
            {/* Role Label */}
            <div className={`text-xs font-medium mb-2 ${isDark ? 'text-white/40' : 'text-gray-400'}`}>
              ShikshaSetu
            </div>
            {/* Shimmer loading bars */}
            <div className="space-y-3 max-w-[300px]" aria-hidden="true">
              <div className={`h-2.5 rounded-full w-full animate-pulse ${isDark ? 'bg-white/10' : 'bg-gray-200'}`} />
              <div className={`h-2.5 rounded-full w-3/4 animate-pulse [animation-delay:150ms] ${isDark ? 'bg-white/10' : 'bg-gray-200'}`} />
              <div className={`h-2.5 rounded-full w-1/2 animate-pulse [animation-delay:300ms] ${isDark ? 'bg-white/10' : 'bg-gray-200'}`} />
            </div>
            <div className={`mt-4 text-xs font-medium animate-pulse ${isDark ? 'text-white/30' : 'text-gray-400'}`}>
              Processing your request...
            </div>
            <span className="sr-only">ShikshaSetu is generating a response, please wait</span>
          </div>
        </div>
      </div>
    </output>
  );
}

/**
 * Indicator shown while regenerating a response
 */
export function RegeneratingIndicator({ isDark, streamingMessage }: RegeneratingIndicatorProps) {
  return (
    <output className="py-6 block" aria-label="Regenerating response">
      <div className="w-full max-w-3xl mx-auto px-4">
        <div className="flex gap-4">
          <div className={`w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center ${isDark ? 'bg-white/10' : 'bg-gray-100'}`}>
            <OmLogo variant="minimal" size={16} color={isDark ? 'dark' : 'light'} animated={false} />
          </div>
          <div className={`flex-1 min-w-0 overflow-hidden prose prose-sm max-w-none pt-1
            ${isDark ? 'prose-invert text-white/90' : 'text-gray-800'}`}>
            <div className="flex items-center gap-2 text-xs font-medium text-orange-500 mb-3 uppercase tracking-wide">
              <RefreshCw className="w-3 h-3 animate-spin" aria-hidden="true" />
              <span>Regenerating Response</span>
            </div>
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
              {streamingMessage}
            </ReactMarkdown>
            <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse rounded-sm" aria-hidden="true" />
          </div>
        </div>
      </div>
    </output>
  );
}
