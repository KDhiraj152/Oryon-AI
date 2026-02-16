/**
 * FileIcon — Memoized file type icon based on MIME type and extension.
 */
import { memo } from 'react';
import { Image, FileText, Mic } from 'lucide-react';
import { AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, SPREADSHEET_EXTENSIONS, getFileExtension } from './chatInputConstants';

export const FileIcon = memo(function FileIcon({ file }: { file: File }) {
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
