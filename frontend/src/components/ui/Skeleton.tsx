import { memo } from 'react';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

/**
 * Base Skeleton component for loading states
 */
export const Skeleton = memo(function Skeleton({
  className = '',
  variant = 'text',
  width,
  height,
  animation = 'pulse',
}: SkeletonProps) {
  const baseClasses = 'bg-[var(--bg-tertiary)]';
  let animationClasses = '';
  if (animation === 'pulse') {
    animationClasses = 'animate-pulse';
  } else if (animation === 'wave') {
    animationClasses = 'animate-shimmer';
  }

  const variantClasses = {
    text: 'rounded-md',
    circular: 'rounded-full',
    rectangular: 'rounded-xl',
  };

  const style: React.CSSProperties = {
    width: width ?? (variant === 'text' ? '100%' : undefined),
    height: height ?? (variant === 'text' ? '1em' : undefined),
  };

  return (
    <div
      className={`${baseClasses} ${animationClasses} ${variantClasses[variant]} ${className}`}
      style={style}
      aria-hidden="true"
    />
  );
});

/**
 * Message skeleton for chat loading states
 */
export const MessageSkeleton = memo(function MessageSkeleton({
  isUser = false,
}: {
  isUser?: boolean;
}) {
  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''} p-4`}>
      {/* Avatar */}
      <Skeleton variant="circular" width={32} height={32} />

      {/* Message content */}
      <div className={`flex-1 max-w-[80%] ${isUser ? 'items-end' : ''}`}>
        <div className="space-y-2">
          <Skeleton variant="text" width="90%" height={16} />
          <Skeleton variant="text" width="75%" height={16} />
          <Skeleton variant="text" width="60%" height={16} />
        </div>
      </div>
    </div>
  );
});

/**
 * Conversation list skeleton
 */
export const ConversationListSkeleton = memo(function ConversationListSkeleton({
  count = 5,
}: {
  count?: number;
}) {
  return (
    <div className="space-y-2 p-2">
      {Array.from({ length: count }).map((_, i) => (
        <div key={`conv-skeleton-${i}`} className="flex items-center gap-3 p-3 rounded-xl">
          <Skeleton variant="circular" width={24} height={24} />
          <div className="flex-1 space-y-2">
            <Skeleton variant="text" width="70%" height={14} />
            <Skeleton variant="text" width="50%" height={12} />
          </div>
        </div>
      ))}
    </div>
  );
});

/**
 * Chat messages list skeleton
 */
export const ChatSkeleton = memo(function ChatSkeleton() {
  return (
    <div className="flex-1 overflow-hidden">
      <div className="space-y-4 p-4">
        <MessageSkeleton isUser />
        <MessageSkeleton />
        <MessageSkeleton isUser />
        <MessageSkeleton />
      </div>
    </div>
  );
});

/**
 * Content card skeleton for landing page
 */
export const ContentCardSkeleton = memo(function ContentCardSkeleton() {
  return (
    <div className="p-6 rounded-2xl border border-[var(--border-color)] bg-[var(--card-bg)]">
      <div className="flex items-start justify-between mb-4">
        <Skeleton variant="circular" width={48} height={48} />
        <Skeleton variant="rectangular" width={60} height={24} />
      </div>
      <Skeleton variant="text" width="60%" height={24} className="mb-2" />
      <div className="space-y-2">
        <Skeleton variant="text" width="100%" height={14} />
        <Skeleton variant="text" width="80%" height={14} />
      </div>
    </div>
  );
});

/**
 * Settings section skeleton
 */
export const SettingsSkeleton = memo(function SettingsSkeleton() {
  return (
    <div className="space-y-8 p-6">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={`settings-section-${i}`} className="space-y-4">
          <Skeleton variant="text" width={120} height={20} />
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, j) => (
              <div key={`settings-row-${i}-${j}`} className="flex items-center justify-between p-4 rounded-xl border border-[var(--border-color)]">
                <div className="flex items-center gap-3">
                  <Skeleton variant="circular" width={40} height={40} />
                  <div className="space-y-1">
                    <Skeleton variant="text" width={100} height={16} />
                    <Skeleton variant="text" width={150} height={12} />
                  </div>
                </div>
                <Skeleton variant="rectangular" width={50} height={28} />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
});

/**
 * Stats card skeleton
 */
export const StatsSkeleton = memo(function StatsSkeleton({
  count = 4,
}: {
  count?: number;
}) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={`stat-skeleton-${i}`} className="p-4 rounded-xl border border-[var(--border-color)] text-center">
          <Skeleton variant="text" width={60} height={12} className="mx-auto mb-2" />
          <Skeleton variant="text" width={80} height={32} className="mx-auto" />
        </div>
      ))}
    </div>
  );
});

/**
 * Full page loading skeleton
 */
export const PageSkeleton = memo(function PageSkeleton() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header skeleton */}
      <div className="h-16 border-b border-[var(--border-color)] flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <Skeleton variant="circular" width={32} height={32} />
          <Skeleton variant="text" width={120} height={20} />
        </div>
        <Skeleton variant="circular" width={36} height={36} />
      </div>

      {/* Content skeleton */}
      <div className="flex-1 p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          <Skeleton variant="text" width="60%" height={32} />
          <div className="space-y-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={`page-line-${i}`} variant="text" width={`${90 - i * 10}%`} height={16} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
});

export default Skeleton;
