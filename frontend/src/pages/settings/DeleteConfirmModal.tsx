import { useEffect, useCallback, useRef } from 'react';
import { Trash2 } from 'lucide-react';

interface DeleteModalProps {
  readonly isOpen: boolean;
  readonly onClose: () => void;
  readonly onConfirm: () => void;
  readonly conversationCount: number;
  readonly isDark: boolean;
}

export function DeleteConfirmModal({
  isOpen,
  onClose,
  onConfirm,
  conversationCount,
  isDark,
}: DeleteModalProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
    } else if (dialog.open) {
      dialog.close();
    }
  }, [isOpen]);

  const handleClose = useCallback(() => {
    onClose();
  }, [onClose]);

  const pluralSuffix = conversationCount === 1 ? '' : 's';

  return (
    <dialog
      ref={dialogRef}
      onClose={handleClose}
      className="fixed inset-0 z-modal flex items-center justify-center p-4 bg-transparent backdrop:bg-black/60 backdrop:backdrop-blur-md m-auto open:flex"
    >
      <div
        className={`w-full max-w-md rounded-modal p-6 shadow-2xl animate-scaleIn
          ${isDark ? 'bg-[#0a0a0a] border border-white/10' : 'bg-white'}`}
      >
        <div
          className={`w-12 h-12 rounded-full flex items-center justify-center mb-4 ${isDark ? 'bg-red-500/10' : 'bg-red-50'}`}
        >
          <Trash2
            className={`w-5 h-5 ${isDark ? 'text-red-400' : 'text-red-600'}`}
            aria-hidden="true"
          />
        </div>

        <h3
          id="delete-modal-title"
          className={`text-title font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}
        >
          Clear All Chat History?
        </h3>
        <p
          className={`text-body-sm mb-6 ${isDark ? 'text-white/60' : 'text-gray-600'}`}
        >
          This will permanently delete all {conversationCount} conversation
          {pluralSuffix}. This action cannot be undone.
        </p>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            autoFocus
            className={`flex-1 px-4 min-h-touch py-2.5 rounded-btn text-body-sm font-medium
              transition-all duration-fast active:scale-[0.98]
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
              ${isDark
                ? 'bg-white/10 text-white hover:bg-white/15 focus-visible:ring-white focus-visible:ring-offset-[#0a0a0a]'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200 focus-visible:ring-gray-400 focus-visible:ring-offset-white'
              }`}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 px-4 min-h-touch py-2.5 rounded-btn text-body-sm font-medium
              bg-red-500 text-white hover:bg-red-600
              transition-all duration-fast active:scale-[0.98]
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0a0a]"
          >
            Delete All
          </button>
        </div>
      </div>
    </dialog>
  );
}
