/**
 * AttachmentsList — Displays file attachments on a message.
 */
import type { Message } from '../../store/chatStore';
import { themed } from './chatHelpers';

export function AttachmentsList({ attachments, isDark }: Readonly<{
  attachments: NonNullable<Message['attachments']>; isDark: boolean;
}>) {
  return (
    <div className="flex flex-wrap gap-2 pt-3">
      {attachments.map((att) => (
        <div key={`${att.name}-${att.url || att.type}`}
          className={`px-3 py-2 rounded-xl text-sm backdrop-blur-md border transition-all duration-200 hover:scale-[1.02]
            ${themed(isDark,
              'bg-white/[0.04] border-white/[0.08] text-white/70 hover:bg-white/[0.08]',
              'bg-gray-50/80 border-gray-200/60 text-gray-600 hover:bg-gray-100/80')}`}>
          <span className="truncate max-w-[200px] block font-medium">{att.name}</span>
        </div>
      ))}
    </div>
  );
}
