import { Message, useThemeStore } from '../../store';
import { User, Copy, RefreshCw, Volume2, VolumeX, ThumbsUp, ThumbsDown, Check, AlertCircle, BookOpen, ChevronDown, ChevronUp, Loader2, ExternalLink } from 'lucide-react';
import ReactMarkdown, { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { useState, useMemo, ReactNode, useCallback, memo } from 'react';
import { OmLogo } from '../landing/OmLogo';

// Audio button label helpers
function getAudioAriaLabel(isPlaying: boolean, isLoading: boolean): string {
  if (isPlaying) return 'Stop audio';
  if (isLoading) return 'Loading audio';
  return 'Read aloud';
}

function getAudioTitle(isPlaying: boolean, isLoading: boolean): string {
  if (isPlaying || isLoading) return 'Click to stop';
  return 'Read aloud';
}

interface ChatMessageProps {
  readonly message: Message;
  readonly isStreaming?: boolean;
  readonly isPlayingAudio?: boolean;
  readonly onRetry?: (messageId: string) => void;
  readonly onCopy?: () => void;
  readonly onAudio?: () => void;
}

interface CodeBlockProps {
  readonly language?: string;
  readonly children: string;
}

// Helper to safely convert children to string
function childrenToString(children: ReactNode): string {
  if (typeof children === 'string') return children;
  if (typeof children === 'number') return String(children);
  if (Array.isArray(children)) return children.map(childrenToString).join('');
  if (children && typeof children === 'object' && 'props' in children) {
    return childrenToString((children as { props: { children?: ReactNode } }).props.children);
  }
  return '';
}

// Memoized CodeBlock component to prevent re-renders
const CodeBlock = memo(function CodeBlock({ language, children }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const { resolvedTheme } = useThemeStore();
  const isDark = resolvedTheme === 'dark';

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [children]);

  // Language display name mapping
  const langDisplayName: Record<string, string> = {
    js: 'JavaScript', javascript: 'JavaScript',
    ts: 'TypeScript', typescript: 'TypeScript',
    py: 'Python', python: 'Python',
    jsx: 'JSX', tsx: 'TSX',
    html: 'HTML', css: 'CSS', scss: 'SCSS',
    json: 'JSON', yaml: 'YAML', yml: 'YAML',
    sql: 'SQL', bash: 'Bash', sh: 'Shell',
    c: 'C', cpp: 'C++', java: 'Java',
    go: 'Go', rust: 'Rust', ruby: 'Ruby',
    php: 'PHP', swift: 'Swift', kotlin: 'Kotlin',
    md: 'Markdown', markdown: 'Markdown',
  };

  return (
    <div className={`rounded-xl overflow-hidden my-5 border font-mono text-[13px] leading-relaxed shadow-sm
      ${isDark ? 'bg-[#1e1e1e] border-white/10' : 'bg-white border-gray-200'}`}>
      <div className={`flex items-center justify-between px-4 py-2.5 border-b
        ${isDark ? 'bg-[#252525] border-white/5' : 'bg-gray-50 border-gray-100'}`}>
        <div className="flex items-center gap-3">
          {/* Minimal window controls */}
          <div className="flex gap-1.5">
            <div className={`w-2.5 h-2.5 rounded-full ${isDark ? 'bg-white/20' : 'bg-gray-300'}`} />
            <div className={`w-2.5 h-2.5 rounded-full ${isDark ? 'bg-white/20' : 'bg-gray-300'}`} />
            <div className={`w-2.5 h-2.5 rounded-full ${isDark ? 'bg-white/20' : 'bg-gray-300'}`} />
          </div>
          <span className={`text-xs font-medium tracking-wide ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
            {langDisplayName[language?.toLowerCase() || ''] || language || 'Code'}
          </span>
        </div>
        <button
          onClick={handleCopy}
          className={`flex items-center gap-1.5 text-xs font-medium transition-all duration-200 px-2 py-1 rounded-md
            ${(() => {
              if (copied) return 'text-emerald-500 bg-emerald-500/10';
              return isDark
                ? 'text-gray-400 hover:text-white hover:bg-white/10'
                : 'text-gray-500 hover:text-gray-900 hover:bg-gray-200';
            })()}`}
        >
          {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <div className="relative group overflow-x-auto p-5">
        <pre className={`m-0 bg-transparent font-mono text-[13px] leading-relaxed ${isDark ? 'text-white' : 'text-gray-800'}`}>
          {children}
        </pre>
      </div>
    </div>
  );
});



// Create markdown components outside the main component to avoid recreation
function createMarkdownComponents(isDark: boolean): Components {
  return {
    // Paragraphs with proper spacing and font
    p: ({ children }) => (
      <p className={`mb-4 last:mb-0 leading-7 text-[15px] font-sans ${isDark ? 'text-white/90' : 'text-gray-700'}`}>
        {children}
      </p>
    ),

    // Lists - clean spacing like ChatGPT
    ul: ({ children }) => (
      <ul className={`list-disc pl-6 my-4 space-y-2 font-sans ${isDark ? 'marker:text-white/50' : 'marker:text-gray-400'}`}>
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className={`list-decimal pl-6 my-4 space-y-2 font-sans ${isDark ? 'marker:text-white/50' : 'marker:text-gray-400'}`}>
        {children}
      </ol>
    ),
    li: ({ children }) => (
      <li className={`leading-relaxed text-[15px] ${isDark ? 'text-white/85' : 'text-gray-700'}`}>
        {children}
      </li>
    ),

    // Text formatting
    strong: ({ children }) => (
      <strong className={`font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
        {children}
      </strong>
    ),
    em: ({ children }) => (
      <em className={`italic ${isDark ? 'text-white/80' : 'text-gray-600'}`}>
        {children}
      </em>
    ),
    del: ({ children }) => (
      <del className={`line-through ${isDark ? 'text-white/50' : 'text-gray-500'}`}>
        {children}
      </del>
    ),

    // Headings with clear hierarchy
    h1: ({ children }) => (
      <h1 className={`text-2xl font-bold mt-8 mb-4 pb-2 border-b font-sans tracking-tight
        ${isDark ? 'text-white border-white/10' : 'text-gray-900 border-gray-200'}`}>
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className={`text-xl font-bold mt-8 mb-4 font-sans tracking-tight ${isDark ? 'text-white' : 'text-gray-900'}`}>
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className={`text-lg font-semibold mt-6 mb-3 font-sans tracking-tight ${isDark ? 'text-white' : 'text-gray-900'}`}>
        {children}
      </h3>
    ),
    h4: ({ children }) => (
      <h4 className={`text-base font-semibold mt-4 mb-2 font-sans ${isDark ? 'text-white/90' : 'text-gray-800'}`}>
        {children}
      </h4>
    ),

    // Code - inline and block
    code: ({ className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || '');
      const content = childrenToString(children);
      const isCodeBlock = className?.includes('language-') || content.includes('\n');

      if (isCodeBlock) {
        return (
          <CodeBlock language={match?.[1]}>
            {content.replace(/\n$/, '')}
          </CodeBlock>
        );
      }
      // Inline code - like ChatGPT
      return (
        <code
          className={`px-1.5 py-0.5 rounded-md text-[13px] font-mono border
            ${isDark
              ? 'bg-white/10 text-orange-200 border-white/10'
              : 'bg-gray-100 text-pink-600 border-gray-200'}`}
          {...props}
        >
          {children}
        </code>
      );
    },
    pre: ({ children }) => <>{children}</>,

    // Blockquote - like Perplexity citations
    blockquote: ({ children }) => (
      <blockquote className={`border-l-4 pl-4 py-2 my-4 rounded-r-lg italic
        ${isDark
          ? 'border-orange-500/50 bg-orange-500/5 text-white/70'
          : 'border-orange-400 bg-orange-50 text-gray-600'}`}>
        {children}
      </blockquote>
    ),

    // Links with icon
    a: ({ href, children }) => (
      <a
        href={href}
        className={`inline-flex items-center gap-1 underline underline-offset-2 transition-colors font-medium
          ${isDark ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-700'}`}
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
        <ExternalLink className="w-3 h-3 opacity-50" />
      </a>
    ),

    // Horizontal rule
    hr: () => (
      <hr className={`my-8 border-0 h-px ${isDark ? 'bg-white/10' : 'bg-gray-200'}`} />
    ),

    // Tables - clean like Notion/ChatGPT
    table: ({ children }) => (
      <div className={`overflow-x-auto my-6 rounded-xl border shadow-sm
        ${isDark ? 'border-white/10 bg-white/[0.02]' : 'border-gray-200 bg-white'}`}>
        <table className={`min-w-full divide-y ${isDark ? 'divide-white/10' : 'divide-gray-200'}`}>
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className={isDark ? 'bg-white/[0.04]' : 'bg-gray-50'}>{children}</thead>
    ),
    th: ({ children }) => (
      <th className={`px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider font-sans
        ${isDark ? 'text-white/60' : 'text-gray-600'}`}>
        {children}
      </th>
    ),
    tbody: ({ children }) => (
      <tbody className={`divide-y ${isDark ? 'divide-white/[0.06]' : 'divide-gray-100'}`}>
        {children}
      </tbody>
    ),
    tr: ({ children }) => (
      <tr className={`transition-colors ${isDark ? 'hover:bg-white/[0.02]' : 'hover:bg-gray-50'}`}>
        {children}
      </tr>
    ),
    td: ({ children }) => (
      <td className={`px-4 py-3 text-sm font-sans ${isDark ? 'text-white/80' : 'text-gray-700'}`}>
        {children}
      </td>
    ),

    // Images - responsive with rounded corners
    img: ({ src, alt }) => (
      <figure className="my-6">
        <img
          src={src}
          alt={alt || ''}
          className={`max-w-full h-auto rounded-xl border
            ${isDark ? 'border-white/10' : 'border-gray-200'}`}
          loading="lazy"
        />
        {alt && (
          <figcaption className={`mt-2 text-center text-sm
            ${isDark ? 'text-white/50' : 'text-gray-500'}`}>
            {alt}
          </figcaption>
        )}
      </figure>
    ),

    // Task lists / checkboxes
    input: ({ type, checked, ...props }) => {
      if (type === 'checkbox') {
        return (
          <input
            type="checkbox"
            checked={checked}
            readOnly
            className={`mr-2 rounded ${isDark ? 'accent-white' : 'accent-black'}`}
            {...props}
          />
        );
      }
      return <input type={type} {...props} />;
    },
  };
}

// Chat message comparison function for React.memo
function arePropsEqual(
  prevProps: ChatMessageProps,
  nextProps: ChatMessageProps
): boolean {
  return (
    prevProps.message.id === nextProps.message.id &&
    prevProps.message.content === nextProps.message.content &&
    prevProps.message.isError === nextProps.message.isError &&
    prevProps.isStreaming === nextProps.isStreaming &&
    prevProps.isPlayingAudio === nextProps.isPlayingAudio
  );
}

// --- Extracted sub-components to reduce cognitive complexity ---

function getThemedClass(isDark: boolean, darkClass: string, lightClass: string): string {
  return isDark ? darkClass : lightClass;
}

/** Avatar icon for message */
function MessageAvatar({ isUser, isDark, isStreaming }: Readonly<{
  isUser: boolean;
  isDark: boolean;
  isStreaming: boolean;
}>) {
  const bg = getThemedClass(isDark, isUser ? 'bg-white/[0.08]' : 'bg-white/[0.06]', 'bg-gray-100');
  const anim = isStreaming ? 'animate-avatar-pop' : '';
  return (
    <div className={`w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center transition-all duration-200 ${bg} ${anim}`}>
      {isUser
        ? <User className={`w-4 h-4 ${getThemedClass(isDark, 'text-white/60', 'text-gray-500')}`} />
        : <OmLogo variant="minimal" size={16} color={isDark ? 'dark' : 'light'} animated={isStreaming} />
      }
    </div>
  );
}

/** Attachments list extracted from ChatMessage */
function AttachmentsList({ attachments, isDark }: Readonly<{
  attachments: NonNullable<Message['attachments']>;
  isDark: boolean;
}>) {
  return (
    <div className="flex flex-wrap gap-2 pt-3">
      {attachments.map((attachment) => (
        <div
          key={`${attachment.name}-${attachment.url || attachment.type}`}
          className={`px-3 py-2 rounded-xl text-sm backdrop-blur-md border transition-all duration-200 hover:scale-[1.02]
            ${getThemedClass(isDark,
              'bg-white/[0.04] border-white/[0.08] text-white/70 hover:bg-white/[0.08]',
              'bg-gray-50/80 border-gray-200/60 text-gray-600 hover:bg-gray-100/80'
            )}`}
        >
          <span className="truncate max-w-[200px] block font-medium">{attachment.name}</span>
        </div>
      ))}
    </div>
  );
}

/** Message body content - user text or AI markdown */
function MessageBody({ message, isDark, isUser, isStreaming, markdownComponents }: Readonly<{
  message: Message;
  isDark: boolean;
  isUser: boolean;
  isStreaming: boolean;
  markdownComponents: ReturnType<typeof createMarkdownComponents>;
}>) {
  if (isUser) {
    return (
      <p className={`whitespace-pre-wrap m-0 text-[15px] leading-relaxed ${getThemedClass(isDark, 'text-white/90', 'text-gray-800')}`}>
        {message.content}
      </p>
    );
  }
  return (
    <>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={markdownComponents}
      >
        {message.content}
      </ReactMarkdown>
      {isStreaming && (
        <span className={`inline-block w-[2px] h-4 ml-0.5 rounded-sm animate-pulse ${getThemedClass(isDark, 'bg-white/70', 'bg-gray-600')}`} />
      )}
    </>
  );
}

/** Metadata footer showing model, latency, tokens */
function MetadataFooter({ message, isDark }: Readonly<{
  message: Message;
  isDark: boolean;
}>) {
  return (
    <div className="pt-3 flex items-center gap-2">
      <div className={`flex items-center gap-3 px-3 py-1.5 rounded-full text-[10px] backdrop-blur-sm
        ${getThemedClass(isDark, 'bg-white/[0.03] text-white/25', 'bg-gray-50 text-gray-400')}`}>
        {message.modelUsed && <span>{message.modelUsed}</span>}
        {message.modelUsed && message.latencyMs && <span className="opacity-50">&bull;</span>}
        {message.latencyMs && <span>{message.latencyMs}ms</span>}
        {message.tokenCount && <><span className="opacity-50">&bull;</span><span>{message.tokenCount} tokens</span></>}
      </div>
    </div>
  );
}

/** Error message display extracted from ChatMessage */
function ErrorMessageView({ message, isDark, onRetry }: Readonly<{
  message: Message;
  isDark: boolean;
  onRetry?: (messageId: string) => void;
}>) {
  return (
    <div className="w-full animate-message-in py-4 [animation-delay:0.05s]">
      <div className="w-full max-w-3xl mx-auto px-4">
        <div className="flex gap-4">
          <div className={`w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center
            ${getThemedClass(isDark, 'bg-red-500/20', 'bg-red-100')}`}>
            <AlertCircle className={`w-4 h-4 ${getThemedClass(isDark, 'text-red-400', 'text-red-500')}`} />
          </div>
          <div className="flex-1 min-w-0 space-y-3 overflow-hidden pt-1">
            <div className={`text-[15px] leading-relaxed break-words ${getThemedClass(isDark, 'text-red-300', 'text-red-600')}`}>
              {message.content}
            </div>
            {onRetry && (
              <button
                onClick={() => onRetry(message.id)}
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-200
                  ${getThemedClass(isDark, 'bg-white/[0.08] hover:bg-white/[0.12] text-white/80', 'bg-gray-100 hover:bg-gray-200 text-gray-700')}`}
              >
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

/** Action buttons extracted from ChatMessage */
function MessageActionButtons({ isDark, copied, onCopy, onRetry, onAudio, isPlayingAudio, isLoadingAudio }: Readonly<{
  isDark: boolean;
  copied: boolean;
  onCopy?: () => void;
  onRetry?: () => void;
  onAudio?: (e: React.MouseEvent) => void;
  isPlayingAudio: boolean;
  isLoadingAudio: boolean;
}>) {
  const inactiveClass = getThemedClass(isDark, 'text-white/30 hover:text-white/70 hover:bg-white/[0.08]', 'text-gray-400 hover:text-gray-600 hover:bg-gray-100');

  return (
    <div
      className="flex items-center gap-1 pt-3 opacity-0 group-hover:opacity-100 transition-all duration-200"
      role="toolbar"
      aria-label="Message actions"
    >
      {onCopy && (
        <button
          onClick={onCopy}
          className={`p-2 rounded-full transition-all duration-200 backdrop-blur-sm ${copied ? 'text-emerald-500 bg-emerald-500/10' : inactiveClass}`}
          aria-label={copied ? 'Copied' : 'Copy'}
          title="Copy"
        >
          {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
        </button>
      )}
      {onRetry && (
        <button onClick={onRetry} className={`p-2 rounded-full transition-all duration-200 backdrop-blur-sm ${inactiveClass}`} aria-label="Regenerate" title="Regenerate">
          <RefreshCw className="w-4 h-4" />
        </button>
      )}
      {onAudio && (
        <button
          onClick={onAudio}
          className={`p-2 rounded-full transition-all duration-200 backdrop-blur-sm ${(() => {
            if (isPlayingAudio || isLoadingAudio) return getThemedClass(isDark, 'text-orange-400 bg-orange-500/15', 'text-orange-500 bg-orange-50');
            return inactiveClass;
          })()}`}
          aria-label={getAudioAriaLabel(isPlayingAudio, isLoadingAudio)}
          title={getAudioTitle(isPlayingAudio, isLoadingAudio)}
        >
          {isLoadingAudio && <Loader2 className="w-4 h-4 animate-spin" />}
          {!isLoadingAudio && isPlayingAudio && <VolumeX className="w-4 h-4" />}
          {!isLoadingAudio && !isPlayingAudio && <Volume2 className="w-4 h-4" />}
        </button>
      )}
      <button className={`p-2 rounded-full transition-all duration-200 backdrop-blur-sm ${inactiveClass}`} aria-label="Good response" title="Good response">
        <ThumbsUp className="w-4 h-4" />
      </button>
      <button className={`p-2 rounded-full transition-all duration-200 backdrop-blur-sm ${inactiveClass}`} aria-label="Bad response" title="Bad response">
        <ThumbsDown className="w-4 h-4" />
      </button>
    </div>
  );
}

/** Citations section extracted from ChatMessage */
function CitationsSection({ message, isDark, showCitations, setShowCitations }: Readonly<{
  message: Message;
  isDark: boolean;
  showCitations: boolean;
  setShowCitations: (v: boolean) => void;
}>) {
  const citations = message.citations!;
  const toggleClass = (() => {
    if (showCitations) return getThemedClass(isDark, 'bg-white/[0.08] text-white/70', 'bg-gray-100 text-gray-700');
    return getThemedClass(isDark, 'text-white/40 hover:text-white/60 hover:bg-white/[0.04]', 'text-gray-400 hover:text-gray-600 hover:bg-gray-50');
  })();

  return (
    <div className="pt-4">
      <button
        onClick={() => setShowCitations(!showCitations)}
        className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium transition-all duration-200 backdrop-blur-sm ${toggleClass}`}
      >
        <BookOpen className="w-4 h-4" />
        <span>{citations.length} source{citations.length > 1 ? 's' : ''}</span>
        {showCitations ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {showCitations && (
        <div className="mt-3 space-y-2 animate-fadeIn">
          {citations.map((citation, idx) => (
            <CitationCard key={citation.id} citation={citation} idx={idx} isDark={isDark} />
          ))}
        </div>
      )}
    </div>
  );
}

function CitationCard({ citation, idx, isDark }: Readonly<{
  citation: NonNullable<Message['citations']>[number];
  idx: number;
  isDark: boolean;
}>) {
  return (
    <div
      className={`p-5 rounded-3xl border backdrop-blur-md transition-all duration-200 hover:scale-[1.01]
        ${getThemedClass(isDark, 'bg-white/[0.03] border-white/[0.06] hover:bg-white/[0.05]', 'bg-gray-50/80 border-gray-100 hover:bg-gray-100/80')}`}
    >
      <div className="flex items-start gap-3">
        <span className={`flex-shrink-0 w-6 h-6 rounded-lg flex items-center justify-center text-xs font-semibold
          ${getThemedClass(isDark, 'bg-white/[0.08] text-white/60', 'bg-gray-200 text-gray-600')}`}>
          {idx + 1}
        </span>
        <div className="flex-1 min-w-0">
          <div className={`font-medium text-sm ${getThemedClass(isDark, 'text-white/80', 'text-gray-700')}`}>
            {citation.title}
          </div>
          {citation.excerpt && (
            <p className={`mt-1.5 text-xs line-clamp-2 leading-relaxed ${getThemedClass(isDark, 'text-white/40', 'text-gray-500')}`}>
              {citation.excerpt}
            </p>
          )}
          <div className={`mt-2 flex items-center gap-3 text-xs ${getThemedClass(isDark, 'text-white/30', 'text-gray-400')}`}>
            <span className={`px-2 py-0.5 rounded-md ${getThemedClass(isDark, 'bg-white/[0.06]', 'bg-gray-100')}`}>
              {Math.round(citation.score * 100)}% match
            </span>
            {citation.url && (
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className={`flex items-center gap-1 hover:underline
                  ${getThemedClass(isDark, 'text-white/50 hover:text-white/70', 'text-gray-500 hover:text-gray-700')}`}
              >
                <ExternalLink className="w-3 h-3" />
                View source
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Custom hook for audio toggle logic */
function useAudioHandler(
  onAudio: (() => void) | undefined,
  isPlayingAudio: boolean,
) {
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);

  const handleAudio = useCallback(async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (isPlayingAudio || isLoadingAudio) {
      setIsLoadingAudio(false);
      onAudio?.();
      return;
    }

    setIsLoadingAudio(true);
    try {
      const result = onAudio?.();
      if (result && typeof (result as Promise<void>).then === 'function') {
        await (result as Promise<void>);
      }
    } catch (error) {
      console.error('Audio playback failed:', error);
    } finally {
      setIsLoadingAudio(false);
    }
  }, [onAudio, isPlayingAudio, isLoadingAudio]);

  return { isLoadingAudio, handleAudio };
}

const ChatMessage = memo(function ChatMessage({
  message,
  isStreaming = false,
  isPlayingAudio = false,
  onRetry,
  onCopy,
  onAudio,
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [showCitations, setShowCitations] = useState(false);
  const isUser = message.role === 'user';
  const isError = message.isError;
  const { resolvedTheme } = useThemeStore();
  const isDark = resolvedTheme === 'dark';
  const hasCitations = message.citations && message.citations.length > 0;
  const hasAttachments = message.attachments && message.attachments.length > 0;

  // Memoize markdown components - must be before any early returns
  const markdownComponents = useMemo(() => createMarkdownComponents(isDark), [isDark]);
  const { isLoadingAudio, handleAudio } = useAudioHandler(onAudio, isPlayingAudio);

  const handleCopy = () => {
    onCopy?.();
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Error message display
  if (isError) {
    return <ErrorMessageView message={message} isDark={isDark} onRetry={onRetry} />;
  }

  // Compute class names
  const animationClass = isStreaming ? 'animate-message-in' : '';
  const showActions = !isUser && !isStreaming;

  return (
    <div
      className={`group w-full py-4 ${animationClass} border-b border-transparent hover:bg-white/[0.02]`}
    >
      <div className="w-full max-w-3xl mx-auto px-4">
        <div className="flex gap-4">
          <MessageAvatar isUser={isUser} isDark={isDark} isStreaming={isStreaming} />

          {/* Content */}
          <div className="flex-1 min-w-0 space-y-0.5 overflow-hidden pt-0.5">
            {/* Role Label */}
            <div className={`text-xs font-medium mb-2 ${getThemedClass(isDark, 'text-white/40', 'text-gray-400')}`}>
              {isUser ? 'You' : 'ShikshaSetu'}
            </div>

            {/* Message Content */}
            <div className={`prose prose-sm max-w-none
              ${isDark ? 'prose-invert' : ''}
              prose-p:leading-7 prose-p:mb-4
              prose-headings:font-semibold
              prose-strong:font-semibold
              prose-ul:my-3 prose-li:my-1`}
            >
              <MessageBody
                message={message}
                isDark={isDark}
                isUser={isUser}
                isStreaming={isStreaming}
                markdownComponents={markdownComponents}
              />
            </div>

            {/* Attachments */}
            {hasAttachments && (
              <AttachmentsList attachments={message.attachments!} isDark={isDark} />
            )}

            {/* Action buttons */}
            {showActions && (
              <MessageActionButtons
                isDark={isDark}
                copied={copied}
                isPlayingAudio={isPlayingAudio}
                isLoadingAudio={isLoadingAudio}
                onCopy={onCopy ? handleCopy : undefined}
                onRetry={onRetry ? () => onRetry(message.id) : undefined}
                onAudio={onAudio ? handleAudio : undefined}
              />
            )}

            {/* Citations */}
            {hasCitations && !isStreaming && (
              <CitationsSection
                message={message}
                isDark={isDark}
                showCitations={showCitations}
                setShowCitations={setShowCitations}
              />
            )}

            {/* Metadata footer */}
            {showActions && (message.modelUsed || message.latencyMs) && (
              <MetadataFooter message={message} isDark={isDark} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}, arePropsEqual);

// Named export for direct imports
export { ChatMessage };

// Default export
export default ChatMessage;
