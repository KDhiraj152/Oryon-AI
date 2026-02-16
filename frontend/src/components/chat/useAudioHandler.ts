/**
 * useAudioHandler — custom hook for audio toggle with loading state.
 */
import { useState, useCallback } from 'react';

export function useAudioHandler(onAudio: (() => void) | undefined, isPlayingAudio: boolean) {
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
    } catch {
      // Audio playback failure is non-critical
    } finally {
      setIsLoadingAudio(false);
    }
  }, [onAudio, isPlayingAudio, isLoadingAudio]);

  return { isLoadingAudio, handleAudio };
}
