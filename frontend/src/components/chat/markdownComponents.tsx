/**
 * Markdown component overrides for react-markdown — theme-aware renderers.
 */
import { ExternalLink } from 'lucide-react';
import type { Components } from 'react-markdown';
import { CodeBlock } from './CodeBlock';
import { childrenToString, themed } from './chatHelpers';

export function createMarkdownComponents(isDark: boolean): Components {
  return {
    p: ({ children }) => (
      <p className={`mb-4 last:mb-0 leading-7 text-[15px] font-sans ${themed(isDark, 'text-white/90', 'text-gray-700')}`}>
        {children}
      </p>
    ),
    ul: ({ children }) => (
      <ul className={`list-disc pl-6 my-4 space-y-2 font-sans ${themed(isDark, 'marker:text-white/50', 'marker:text-gray-400')}`}>
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className={`list-decimal pl-6 my-4 space-y-2 font-sans ${themed(isDark, 'marker:text-white/50', 'marker:text-gray-400')}`}>
        {children}
      </ol>
    ),
    li: ({ children }) => (
      <li className={`leading-relaxed text-[15px] ${themed(isDark, 'text-white/85', 'text-gray-700')}`}>
        {children}
      </li>
    ),
    strong: ({ children }) => (
      <strong className={`font-semibold ${themed(isDark, 'text-white', 'text-gray-900')}`}>{children}</strong>
    ),
    em: ({ children }) => (
      <em className={`italic ${themed(isDark, 'text-white/80', 'text-gray-600')}`}>{children}</em>
    ),
    del: ({ children }) => (
      <del className={`line-through ${themed(isDark, 'text-white/50', 'text-gray-500')}`}>{children}</del>
    ),
    h1: ({ children }) => (
      <h1 className={`text-2xl font-bold mt-8 mb-4 pb-2 border-b font-sans tracking-tight
        ${themed(isDark, 'text-white border-white/10', 'text-gray-900 border-gray-200')}`}>
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className={`text-xl font-bold mt-8 mb-4 font-sans tracking-tight ${themed(isDark, 'text-white', 'text-gray-900')}`}>
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className={`text-lg font-semibold mt-6 mb-3 font-sans tracking-tight ${themed(isDark, 'text-white', 'text-gray-900')}`}>
        {children}
      </h3>
    ),
    h4: ({ children }) => (
      <h4 className={`text-base font-semibold mt-4 mb-2 font-sans ${themed(isDark, 'text-white/90', 'text-gray-800')}`}>
        {children}
      </h4>
    ),
    code: ({ className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || '');
      const content = childrenToString(children);
      const isCodeBlock = className?.includes('language-') || content.includes('\n');

      if (isCodeBlock) {
        return <CodeBlock language={match?.[1]}>{content.replace(/\n$/, '')}</CodeBlock>;
      }
      return (
        <code
          className={`px-1.5 py-0.5 rounded-md text-[13px] font-mono border
            ${themed(isDark,
              'bg-white/10 text-orange-200 border-white/10',
              'bg-gray-100 text-pink-600 border-gray-200')}`}
          {...props}
        >
          {children}
        </code>
      );
    },
    pre: ({ children }) => <>{children}</>,
    blockquote: ({ children }) => (
      <blockquote className={`border-l-4 pl-4 py-2 my-4 rounded-r-lg italic
        ${themed(isDark,
          'border-orange-500/50 bg-orange-500/5 text-white/70',
          'border-orange-400 bg-orange-50 text-gray-600')}`}>
        {children}
      </blockquote>
    ),
    a: ({ href, children }) => (
      <a href={href} className={`inline-flex items-center gap-1 underline underline-offset-2 transition-colors font-medium
        ${themed(isDark, 'text-blue-400 hover:text-blue-300', 'text-blue-600 hover:text-blue-700')}`}
        target="_blank" rel="noopener noreferrer">
        {children}
        <ExternalLink className="w-3 h-3 opacity-50" />
      </a>
    ),
    hr: () => <hr className={`my-8 border-0 h-px ${themed(isDark, 'bg-white/10', 'bg-gray-200')}`} />,
    table: ({ children }) => (
      <div className={`overflow-x-auto my-6 rounded-xl border shadow-sm
        ${themed(isDark, 'border-white/10 bg-white/[0.02]', 'border-gray-200 bg-white')}`}>
        <table className={`min-w-full divide-y ${themed(isDark, 'divide-white/10', 'divide-gray-200')}`}>
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className={themed(isDark, 'bg-white/[0.04]', 'bg-gray-50')}>{children}</thead>
    ),
    th: ({ children }) => (
      <th className={`px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider font-sans
        ${themed(isDark, 'text-white/60', 'text-gray-600')}`}>
        {children}
      </th>
    ),
    tbody: ({ children }) => (
      <tbody className={`divide-y ${themed(isDark, 'divide-white/[0.06]', 'divide-gray-100')}`}>
        {children}
      </tbody>
    ),
    tr: ({ children }) => (
      <tr className={`transition-colors ${themed(isDark, 'hover:bg-white/[0.02]', 'hover:bg-gray-50')}`}>
        {children}
      </tr>
    ),
    td: ({ children }) => (
      <td className={`px-4 py-3 text-sm font-sans ${themed(isDark, 'text-white/80', 'text-gray-700')}`}>
        {children}
      </td>
    ),
    img: ({ src, alt }) => (
      <figure className="my-6">
        <img src={src} alt={alt || ''} className={`max-w-full h-auto rounded-xl border
          ${themed(isDark, 'border-white/10', 'border-gray-200')}`} loading="lazy" />
        {alt && (
          <figcaption className={`mt-2 text-center text-sm ${themed(isDark, 'text-white/50', 'text-gray-500')}`}>
            {alt}
          </figcaption>
        )}
      </figure>
    ),
    input: ({ type, checked, ...props }) => {
      if (type === 'checkbox') {
        return <input type="checkbox" checked={checked} readOnly
          className={`mr-2 rounded ${themed(isDark, 'accent-white', 'accent-black')}`} {...props} />;
      }
      return <input type={type} {...props} />;
    },
  };
}
