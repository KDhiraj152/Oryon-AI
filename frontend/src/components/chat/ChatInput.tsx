import { useState, useRef, useEffect, useCallback, KeyboardEvent, memo } from 'react';
import { Paperclip, X, Image, FileText, Globe, Check, ArrowUp, Mic, Loader2 } from 'lucide-react';
import { useThemeStore, useAuthStore } from '../../store';
import { audio as audioApi } from '../../api';

const SUPPORTED_LANGUAGES = [
  { code: 'auto', name: 'Auto Detect', native: 'Auto' },
  { code: 'en', name: 'English', native: 'English' },
  { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
  { code: 'bn', name: 'Bengali', native: 'বাংলা' },
  { code: 'te', name: 'Telugu', native: 'తెలుగు' },
  { code: 'ta', name: 'Tamil', native: 'தமிழ்' },
  { code: 'mr', name: 'Marathi', native: 'मराठी' },
  { code: 'gu', name: 'Gujarati', native: 'ગુજરાતી' },
  { code: 'kn', name: 'Kannada', native: 'ಕನ್ನಡ' },
  { code: 'ml', name: 'Malayalam', native: 'മലയാളം' },
  { code: 'pa', name: 'Punjabi', native: 'ਪੰਜਾਬੀ' },
  { code: 'or', name: 'Odia', native: 'ଓଡ଼ିଆ' },
];

// File extension to icon mapping
const AUDIO_EXTENSIONS = new Set(['.mp3', '.wav', '.m4a', '.ogg', '.flac']);
const VIDEO_EXTENSIONS = new Set(['.mp4', '.webm', '.mov', '.avi', '.mkv']);
const SPREADSHEET_EXTENSIONS = new Set(['.csv', '.xls', '.xlsx']);

function getFileExtension(filename: string): string {
  return filename.toLowerCase().slice(filename.lastIndexOf('.'));
}

// Memoized file icon component
const FileIcon = memo(function FileIcon({ file }: { file: File }) {
  const ext = getFileExtension(file.name);

  if (file.type.startsWith('image/')) {
    return <Image className="w-4 h-4 text-blue-500" />;
  }
  if (file.type.startsWith('audio/') || AUDIO_EXTENSIONS.has(ext)) {
    return <Mic className="w-4 h-4 text-purple-500" />;
  }
  if (file.type.startsWith('video/') || VIDEO_EXTENSIONS.has(ext)) {
    return <FileText className="w-4 h-4 text-red-500" />;
  }
  if (ext === '.pdf') {
    return <FileText className="w-4 h-4 text-red-600" />;
  }
  if (SPREADSHEET_EXTENSIONS.has(ext)) {
    return <FileText className="w-4 h-4 text-green-600" />;
  }
  return <FileText className="w-4 h-4" />;
});

interface ChatInputProps {
  readonly onSend: (message: string, files?: File[], language?: string) => void;
  readonly disabled?: boolean;
  readonly placeholder?: string;
  readonly selectedLanguage?: string;
  readonly onLanguageChange?: (lang: string) => void;
  readonly onError?: (error: string) => void;
}

const ChatInput = memo(function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Ask anything — code, ideas, research, or just chat...',
  selectedLanguage = 'auto',
  onLanguageChange,
  onError,
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [showLanguageMenu, setShowLanguageMenu] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const languageMenuRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const { resolvedTheme } = useThemeStore();
  const { isAuthenticated } = useAuthStore();
  const isDark = resolvedTheme === 'dark';

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (languageMenuRef.current && !languageMenuRef.current.contains(e.target as Node)) {
        setShowLanguageMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Focus textarea on mount
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyboard = (e: globalThis.KeyboardEvent) => {
      // Cmd/Ctrl + K to focus input
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        textareaRef.current?.focus();
      }
    };

    document.addEventListener('keydown', handleKeyboard);
    return () => document.removeEventListener('keydown', handleKeyboard);
  }, []);

  const handleSubmit = () => {
    if ((message.trim() || files.length > 0) && !disabled) {
      onSend(message.trim(), files.length > 0 ? files : undefined, selectedLanguage);
      setMessage('');
      setFiles([]);
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      setFiles(prev => [...prev, ...newFiles]);
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    if (droppedFiles.length > 0) {
      setFiles(prev => [...prev, ...droppedFiles]);
    }
  };

  // Auto-resize textarea
  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, 200);
    textarea.style.height = `${newHeight}px`;
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [message, adjustHeight]);

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Voice Recording Functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4'
      });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        for (const track of stream.getTracks()) {
          track.stop();
        }

        if (audioChunksRef.current.length === 0) return;

        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType
        });

        // Transcribe audio using V2 STT API (Whisper V3 Turbo)
        // Use guest endpoint for unauthenticated users
        setIsTranscribing(true);
        try {
          const langCode = selectedLanguage === 'auto' ? 'auto' : selectedLanguage;
          const result = isAuthenticated
            ? await audioApi.speechToText(audioBlob, langCode)
            : await audioApi.speechToTextGuest(audioBlob, langCode);

          if (result.text) {
            setMessage(prev => prev + (prev ? ' ' : '') + result.text);
            textareaRef.current?.focus();
          }
        } catch (error) {
          console.error('Transcription failed:', error);
          onError?.('Voice transcription failed. Please try again or type your message.');
        } finally {
          setIsTranscribing(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Failed to start recording:', error);
      onError?.('Microphone access denied. Please enable microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  return (
    <div className="w-full relative">
      {/* Language Menu - Floating above */}
      {showLanguageMenu && (
        <div
          ref={languageMenuRef}
          className={`absolute bottom-full mb-4 left-4 sm:left-auto z-50 w-64 max-h-80 overflow-y-auto rounded-3xl shadow-2xl border backdrop-blur-xl
            ${isDark
              ? 'bg-[#1a1a1a]/90 border-white/10 text-white'
              : 'bg-white/90 border-gray-200 text-gray-900'}`}
        >
          <div className="p-2 grid gap-1">
            {SUPPORTED_LANGUAGES.map((lang) => (
              <button
                key={lang.code}
                onClick={() => {
                  onLanguageChange?.(lang.code);
                  setShowLanguageMenu(false);
                }}
                className={`flex items-center justify-between w-full px-4 py-3 rounded-2xl text-sm transition-all duration-200
                  ${(() => {
                    if (selectedLanguage === lang.code) return isDark ? 'bg-white/10 text-white' : 'bg-black/5 text-black font-medium';
                    return isDark ? 'hover:bg-white/5 text-white/70 hover:text-white' : 'hover:bg-black/5 text-gray-600 hover:text-black';
                  })()}`}
              >
                <div className="flex flex-col items-start">
                  <span className="font-medium">{lang.name}</span>
                  <span className={`text-xs ${isDark ? 'text-white/40' : 'text-gray-400'}`}>{lang.native}</span>
                </div>
                {selectedLanguage === lang.code && <Check className="w-4 h-4" />}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main Input Container */}
      <div
        className={`relative group transition-all duration-300 ease-out-expo
          ${isDragOver ? 'scale-[1.02]' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Drag Overlay */}
        {isDragOver && (
          <div className={`absolute inset-0 z-50 rounded-[26px] border-2 border-dashed flex items-center justify-center backdrop-blur-sm
            ${isDark ? 'border-white/20 bg-black/60' : 'border-black/20 bg-white/60'}`}>
            <div className="pointer-events-none flex flex-col items-center gap-2 animate-bounce">
              <Paperclip className="w-8 h-8 opacity-50" />
              <span className="text-sm font-medium opacity-70">Drop files here</span>
            </div>
          </div>
        )}

        {/* Input Capsule */}
        <div className={`relative flex flex-col rounded-[26px] shadow-2xl transition-colors duration-300
          ${isDark
            ? 'bg-[#1a1a1a]/60 border border-white/[0.08] shadow-black/20'
            : 'bg-white/60 border border-black/[0.08] shadow-xl shadow-black/5'}
          backdrop-blur-xl`}
        >
          {/* File Previews */}
          {files.length > 0 && (
            <div className="flex flex-wrap gap-2 px-3 pt-3 pb-1">
              {files.map((file, i) => (
                <div
                  key={`file-${file.name}-${file.size}`}
                  className={`group relative flex items-center gap-2 pl-3 pr-8 py-1.5 rounded-xl text-xs font-medium transition-all duration-200 border
                    ${isDark
                      ? 'bg-white/5 border-white/10 text-white hover:bg-white/10'
                      : 'bg-gray-50 border-gray-200 text-gray-900 hover:bg-gray-100'}`}
                >
                  <FileIcon file={file} />
                  <span className="max-w-[120px] truncate">{file.name}</span>
                  <button
                    onClick={() => removeFile(i)}
                    className={`absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded-full opacity-0 group-hover:opacity-100 transition-all
                      ${isDark ? 'hover:bg-white/20 text-white/60' : 'hover:bg-black/10 text-black/60'}`}
                    title="Remove file"
                    aria-label={`Remove ${file.name}`}
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Text Area & Controls */}
          <div className="flex items-center gap-1 p-2">
            {/* Left Actions */}
            <div className="flex items-center">
              <input
                type="file"
                multiple
                className="hidden"
                ref={fileInputRef}
                onChange={handleFileSelect}
                title="Select files to attach"
                aria-label="Select files to attach"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className={`p-2.5 rounded-full transition-all duration-200
                  ${isDark
                    ? 'text-white/40 hover:text-white/70 hover:bg-white/[0.06]'
                    : 'text-black/30 hover:text-black/60 hover:bg-black/[0.04]'}`}
                title="Attach files"
              >
                <Paperclip className="w-5 h-5" />
              </button>

              <button
                onClick={() => setShowLanguageMenu(!showLanguageMenu)}
                className={`hidden sm:flex items-center gap-1.5 px-3 py-2 rounded-full text-[13px] font-medium transition-all duration-200
                  ${isDark
                    ? 'text-white/40 hover:text-white/70 hover:bg-white/[0.06]'
                    : 'text-black/30 hover:text-black/60 hover:bg-black/[0.04]'}`}
                title="Change language"
              >
                <Globe className="w-4 h-4" />
                <span>{SUPPORTED_LANGUAGES.find(l => l.code === selectedLanguage)?.name?.split(' ')[0] || 'Auto'}</span>
              </button>
            </div>

            {/* Text Input */}
            <textarea
              ref={textareaRef}
              value={message}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder={isRecording ? "Listening..." : placeholder}
              disabled={disabled || isRecording}
              rows={1}
              spellCheck={false}
              className={`flex-1 max-h-[200px] min-h-[44px] py-2.5 px-3 bg-transparent border-0 focus:ring-0 focus:outline-none resize-none text-[15px] font-medium leading-relaxed scrollbar-hide
                placeholder:font-normal placeholder:transition-opacity duration-200
                ${isDark
                  ? 'text-white placeholder:text-white/30 caret-white'
                  : 'text-gray-900 placeholder:text-gray-400 caret-black'}
                ${isRecording ? 'placeholder:animate-pulse' : ''}`}
            />

            {/* Right Actions */}
            <div className="flex items-center gap-1">
              {/* Voice Input */}
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={disabled || isTranscribing}
                className={`p-2.5 rounded-full transition-all duration-300 relative
                  ${(() => {
                    if (isRecording) return 'bg-red-500/90 text-white';
                    return isDark
                      ? 'text-white/40 hover:text-white/70 hover:bg-white/[0.06]'
                      : 'text-black/30 hover:text-black/60 hover:bg-black/[0.04]';
                  })()}`}
                title={isRecording ? "Stop recording" : "Start voice input"}
              >
                {isTranscribing ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Mic className="w-5 h-5" />
                )}
              </button>

              {/* Send Button */}
              <button
                onClick={handleSubmit}
                disabled={(!message.trim() && files.length === 0) || disabled || isRecording}
                className={`p-2.5 rounded-full transition-all duration-200 flex items-center justify-center
                  ${(() => {
                    if ((!message.trim() && files.length === 0) || disabled)
                      return isDark ? 'bg-white/[0.06] text-white/20' : 'bg-black/[0.04] text-black/20';
                    return isDark
                      ? 'bg-white text-black hover:bg-white/90'
                      : 'bg-black text-white hover:bg-black/90';
                  })()}`}
                title="Send message"
                aria-label="Send message"
              >
                <ArrowUp className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

export default ChatInput;
