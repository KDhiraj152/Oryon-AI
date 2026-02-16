/**
 * Shared helper utilities for chat components.
 */
import type { ReactNode } from 'react';

/** Recursively extract text content from React children */
export function childrenToString(children: ReactNode): string {
  if (typeof children === 'string') return children;
  if (typeof children === 'number') return String(children);
  if (Array.isArray(children)) return children.map(childrenToString).join('');
  if (children && typeof children === 'object' && 'props' in children) {
    return childrenToString((children as { props: { children?: ReactNode } }).props.children);
  }
  return '';
}

/** Return dark or light value based on theme */
export function themed(isDark: boolean, dark: string, light: string): string {
  return isDark ? dark : light;
}

/** Accessible label for audio button */
export function getAudioAriaLabel(isPlaying: boolean, isLoading: boolean): string {
  if (isPlaying) return 'Stop audio';
  if (isLoading) return 'Loading audio';
  return 'Read aloud';
}

/** Tooltip for audio button */
export function getAudioTitle(isPlaying: boolean, isLoading: boolean): string {
  if (isPlaying || isLoading) return 'Click to stop';
  return 'Read aloud';
}
