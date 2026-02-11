import { useState } from 'react';
import { X, Download, FileText, FileJson, File, Check, Loader2 } from 'lucide-react';
import { useThemeStore, Message } from '../../store';
import { aiCore } from '../../api';

function triggerFileDownload(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

interface ExportModalProps {
  readonly isOpen: boolean;
  readonly onClose: () => void;
  readonly messages: Message[];
  readonly conversationTitle?: string;
}

type ExportFormat = 'markdown' | 'json' | 'text';

const formatOptions: Array<{
  id: ExportFormat;
  name: string;
  description: string;
  icon: typeof FileText;
  extension: string;
}> = [
  {
    id: 'markdown',
    name: 'Markdown',
    description: 'Formatted with headers and code blocks',
    icon: FileText,
    extension: '.md',
  },
  {
    id: 'json',
    name: 'JSON',
    description: 'Structured data with metadata',
    icon: FileJson,
    extension: '.json',
  },
  {
    id: 'text',
    name: 'Plain Text',
    description: 'Simple text format',
    icon: File,
    extension: '.txt',
  },
];

export default function ExportModal({ isOpen, onClose, messages, conversationTitle }: ExportModalProps) {
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('markdown');
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [includeCitations, setIncludeCitations] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);
  const { resolvedTheme } = useThemeStore();
  const isDark = resolvedTheme === 'dark';

  if (!isOpen) return null;

  const handleExport = async () => {
    setIsExporting(true);
    setExportSuccess(false);

    try {
      // Convert messages to export format
      const conversation = messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp,
        citations: msg.citations,
      }));

      // Try API export first, fallback to local
      let content: string;
      try {
        content = await aiCore.exportConversation(conversation, {
          format: selectedFormat,
          include_metadata: includeMetadata,
          include_citations: includeCitations,
        });
      } catch {
        // Fallback to local export
        content = localExport(messages, selectedFormat, includeMetadata);
      }

      const formatInfo = formatOptions.find((f) => f.id === selectedFormat)!;
      const filename = `${conversationTitle || 'conversation'}-${new Date().toISOString().split('T')[0]}${formatInfo.extension}`;
      const mimeType = selectedFormat === 'json' ? 'application/json' : 'text/plain';
      triggerFileDownload(content, filename, mimeType);

      setExportSuccess(true);
      setTimeout(() => {
        onClose();
        setExportSuccess(false);
      }, 1500);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-50 backdrop-blur-sm animate-fadeIn"
        onClick={onClose}
        onKeyDown={(e) => { if (e.key === 'Escape') onClose(); }}
        role="none"
      />

      {/* Modal */}
      <div className={`fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-md
        ${isDark ? 'bg-[#141414]' : 'bg-white'} rounded-2xl shadow-2xl animate-scaleIn`}>

        {/* Header */}
        <div className={`flex items-center justify-between p-4 border-b
          ${isDark ? 'border-white/[0.06]' : 'border-gray-100'}`}>
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${isDark ? 'bg-white/5' : 'bg-gray-100'}`}>
              <Download className={`w-5 h-5 ${isDark ? 'text-white/70' : 'text-gray-600'}`} />
            </div>
            <div>
              <h2 className={`text-base font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Export Conversation
              </h2>
              <p className={`text-xs ${isDark ? 'text-white/40' : 'text-gray-500'}`}>
                {messages.length} messages
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className={`p-2 rounded-lg transition-colors
              ${isDark ? 'hover:bg-white/5 text-white/40' : 'hover:bg-gray-100 text-gray-400'}`}
            title="Close export modal"
            aria-label="Close export modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Format Selection */}
          <div className="space-y-2">
            <span id="export-format-label" className={`text-xs font-medium ${isDark ? 'text-white/60' : 'text-gray-600'}`}>
              Export Format
            </span>
            <div className="grid grid-cols-3 gap-2" role="radiogroup" aria-labelledby="export-format-label">
              {formatOptions.map((format) => {
                const isSelected = selectedFormat === format.id;
                let borderClass: string;
                let iconClass: string;
                let textClass: string;
                if (isSelected) {
                  borderClass = isDark ? 'border-white/20 bg-white/5' : 'border-gray-300 bg-gray-50';
                  iconClass = isDark ? 'text-white' : 'text-gray-900';
                  textClass = isDark ? 'text-white' : 'text-gray-900';
                } else {
                  borderClass = isDark ? 'border-white/[0.06] hover:border-white/10' : 'border-gray-100 hover:border-gray-200';
                  iconClass = isDark ? 'text-white/40' : 'text-gray-400';
                  textClass = isDark ? 'text-white/60' : 'text-gray-600';
                }
                return (
                  <button
                    key={format.id}
                    onClick={() => setSelectedFormat(format.id)}
                    className={`flex flex-col items-center gap-2 p-3 rounded-2xl border transition-all
                      ${borderClass}`}
                  >
                    <format.icon className={`w-5 h-5 ${iconClass}`} />
                    <span className={`text-xs font-medium ${textClass}`}>
                      {format.name}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Options */}
          <div className="space-y-3">
            <span className={`text-xs font-medium ${isDark ? 'text-white/60' : 'text-gray-600'}`}>
              Options
            </span>

            {/* Include Metadata */}
            <label className={`flex items-center justify-between p-3 rounded-2xl border cursor-pointer
              ${isDark ? 'border-white/[0.06] hover:border-white/10' : 'border-gray-100 hover:border-gray-200'}`}>
              <span className={`text-sm ${isDark ? 'text-white/80' : 'text-gray-700'}`}>
                Include metadata
              </span>
              <div className={`w-10 h-6 rounded-full p-0.5 transition-colors ${(() => {
                if (includeMetadata) return isDark ? 'bg-white' : 'bg-gray-900';
                return isDark ? 'bg-white/10' : 'bg-gray-200';
              })()}`}>
                <div className={`w-5 h-5 rounded-full transition-transform ${(() => {
                  if (includeMetadata) return 'translate-x-4 bg-black';
                  return isDark ? 'bg-white/40' : 'bg-white';
                })()}`} />
              </div>
              <input
                type="checkbox"
                checked={includeMetadata}
                onChange={(e) => setIncludeMetadata(e.target.checked)}
                className="sr-only"
              />
            </label>

            {/* Include Citations */}
            <label className={`flex items-center justify-between p-3 rounded-2xl border cursor-pointer
              ${isDark ? 'border-white/[0.06] hover:border-white/10' : 'border-gray-100 hover:border-gray-200'}`}>
              <span className={`text-sm ${isDark ? 'text-white/80' : 'text-gray-700'}`}>
                Include citations
              </span>
              <div className={`w-10 h-6 rounded-full p-0.5 transition-colors ${(() => {
                if (includeCitations) return isDark ? 'bg-white' : 'bg-gray-900';
                return isDark ? 'bg-white/10' : 'bg-gray-200';
              })()}`}>
                <div className={`w-5 h-5 rounded-full transition-transform ${(() => {
                  if (includeCitations) return 'translate-x-4 bg-black';
                  return isDark ? 'bg-white/40' : 'bg-white';
                })()}`} />
              </div>
              <input
                type="checkbox"
                checked={includeCitations}
                onChange={(e) => setIncludeCitations(e.target.checked)}
                className="sr-only"
              />
            </label>
          </div>
        </div>

        {/* Footer */}
        <div className={`p-4 border-t ${isDark ? 'border-white/[0.06]' : 'border-gray-100'}`}>
          <button
            onClick={handleExport}
            disabled={isExporting}
            className={`w-full py-3 rounded-full text-sm font-medium flex items-center justify-center gap-2 transition-all
              ${isDark
                ? 'bg-white text-black hover:bg-gray-100 disabled:bg-white/50'
                : 'bg-gray-900 text-white hover:bg-gray-800 disabled:bg-gray-400'
              }`}
          >
            {(() => {
              if (isExporting) return <Loader2 className="w-4 h-4 animate-spin" />;
              if (exportSuccess) return (<><Check className="w-4 h-4" /> Exported!</>);
              return (<><Download className="w-4 h-4" /> Export</>);
            })()}
          </button>
        </div>
      </div>
    </>
  );
}

// Local fallback export function
function localExport(messages: Message[], format: ExportFormat, includeMetadata: boolean): string {
  const timestamp = new Date().toISOString();

  if (format === 'json') {
    const data = {
      exported_at: timestamp,
      message_count: messages.length,
      messages: messages.map((m) => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
        ...(m.citations && { citations: m.citations }),
        ...(includeMetadata && m.modelUsed && { model: m.modelUsed }),
        ...(includeMetadata && m.latencyMs && { latency_ms: m.latencyMs }),
      })),
    };
    return JSON.stringify(data, null, 2);
  }

  if (format === 'markdown') {
    let md = `# Conversation Export\n\n`;
    md += `*Exported on ${new Date(timestamp).toLocaleString()}*\n\n`;
    md += `---\n\n`;

    messages.forEach((m) => {
      md += `## ${m.role === 'user' ? '👤 You' : '🤖 Assistant'}\n\n`;
      md += `${m.content}\n\n`;
      if (m.citations && m.citations.length > 0) {
        md += `**Sources:**\n`;
        m.citations.forEach((c, i) => {
          md += `${i + 1}. ${c.title}${c.url ? ' - [Link](' + c.url + ')' : ''}\n`;
        });
        md += '\n';
      }
      md += `---\n\n`;
    });

    return md;
  }

  // Plain text
  let text = `Conversation Export - ${new Date(timestamp).toLocaleString()}\n\n`;
  text += '='.repeat(50) + '\n\n';

  messages.forEach((m) => {
    text += `[${m.role === 'user' ? 'You' : 'Assistant'}]\n`;
    text += `${m.content}\n\n`;
    text += '-'.repeat(30) + '\n\n';
  });

  return text;
}
