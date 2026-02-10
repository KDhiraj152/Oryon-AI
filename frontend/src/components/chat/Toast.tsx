import { useEffect } from 'react';
import { AlertCircle, X } from 'lucide-react';

export type ToastType = 'error' | 'success' | 'info';

export interface ToastProps {
  readonly message: string;
  readonly type: ToastType;
  readonly onClose: () => void;
}

const getToastBgColor = (type: ToastType): string => {
  switch (type) {
    case 'error': return 'bg-red-500/90';
    case 'success': return 'bg-emerald-500/90';
    default: return 'bg-gray-800/90';
  }
};

export function Toast({ message, type, onClose }: ToastProps) {
  useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = getToastBgColor(type);

  return (
    <div className={`fixed top-4 right-4 z-50 flex items-center gap-3 px-4 py-3 rounded-full shadow-xl backdrop-blur-md ${bgColor} text-white animate-slideIn`}>
      <AlertCircle className="w-5 h-5 flex-shrink-0" />
      <span className="text-sm font-medium">{message}</span>
      <button onClick={onClose} className="p-1 hover:bg-white/20 rounded-full transition-colors" title="Dismiss notification" aria-label="Dismiss notification">
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}
