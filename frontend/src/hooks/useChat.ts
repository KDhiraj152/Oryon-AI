import { useState, useCallback, useRef, useEffect } from 'react';
import { audio as audioApi } from '../api';
import type { ToastType } from '../components/ui/Toast';

/**
 * Language detection based on Unicode script ranges
 */
export function detectLanguage(text: string): string {
  if (/[\u0900-\u097F]/.test(text)) return 'hi'; // Devanagari
  if (/[\u0C00-\u0C7F]/.test(text)) return 'te'; // Telugu
  if (/[\u0B80-\u0BFF]/.test(text)) return 'ta'; // Tamil
  if (/[\u0C80-\u0CFF]/.test(text)) return 'kn'; // Kannada
  if (/[\u0D00-\u0D7F]/.test(text)) return 'ml'; // Malayalam
  if (/[\u0980-\u09FF]/.test(text)) return 'bn'; // Bengali
  if (/[\u0A80-\u0AFF]/.test(text)) return 'gu'; // Gujarati
  if (/[\u0A00-\u0A7F]/.test(text)) return 'pa'; // Punjabi
  if (/[\u0B00-\u0B7F]/.test(text)) return 'or'; // Odia
  return 'en';
}

interface UseAudioPlaybackOptions {
  selectedLanguage: string;
  showToast: (message: string, type: ToastType) => void;
}

interface UseAudioPlaybackReturn {
  playingAudioMessageId: string | null;
  handleAudio: (content: string, messageId: string) => Promise<void>;
  stopAudio: () => void;
}

/**
 * Custom hook for managing audio playback with TTS fallback
 */
export function useAudioPlayback({ selectedLanguage, showToast }: UseAudioPlaybackOptions): UseAudioPlaybackReturn {
  const [playingAudioMessageId, setPlayingAudioMessageId] = useState<string | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  const stopAudio = useCallback(() => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }
    if ('speechSynthesis' in globalThis) {
      globalThis.speechSynthesis.cancel();
    }
    setPlayingAudioMessageId(null);
  }, []);

  const handleAudio = useCallback(async (content: string, messageId: string): Promise<void> => {
    // Toggle behavior
    if (playingAudioMessageId === messageId) {
      stopAudio();
      return;
    }

    stopAudio();
    setPlayingAudioMessageId(messageId);

    try {
      const langForTTS = selectedLanguage === 'auto'
        ? detectLanguage(content)
        : selectedLanguage;

      const data = await audioApi.textToSpeechGuest(content, langForTTS, 'female');

      if (data.success && data.audio_data) {
        return new Promise((resolve) => {
          const audioElement = audioApi.playBase64Audio(data.audio_data!, data.audio_format || 'audio/mpeg');
          currentAudioRef.current = audioElement;
          audioElement.onended = () => {
            currentAudioRef.current = null;
            setPlayingAudioMessageId(null);
            resolve();
          };
          audioElement.onerror = () => {
            currentAudioRef.current = null;
            setPlayingAudioMessageId(null);
            resolve();
          };
        });
      }

      // Browser TTS fallback
      if (data.use_browser_tts && 'speechSynthesis' in globalThis) {
        return new Promise((resolve) => {
          globalThis.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(content);
          utterance.rate = 0.9;
          utterance.pitch = 1;
          utterance.onend = () => {
            setPlayingAudioMessageId(null);
            resolve();
          };
          utterance.onerror = () => {
            setPlayingAudioMessageId(null);
            resolve();
          };
          globalThis.speechSynthesis.speak(utterance);
          showToast('Using browser voice (server TTS unavailable)', 'info');
        });
      }
    } catch (error) {
      console.warn('Backend TTS failed, using browser fallback:', error);
      setPlayingAudioMessageId(null);

      if ('speechSynthesis' in globalThis) {
        return new Promise((resolve) => {
          setPlayingAudioMessageId(messageId);
          globalThis.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(content);
          utterance.rate = 0.9;
          utterance.pitch = 1;
          utterance.onend = () => {
            setPlayingAudioMessageId(null);
            resolve();
          };
          utterance.onerror = () => {
            setPlayingAudioMessageId(null);
            resolve();
          };
          globalThis.speechSynthesis.speak(utterance);
          showToast('Using browser voice (server TTS unavailable)', 'info');
        });
      } else {
        showToast('Audio playback not available in this browser', 'error');
      }
    }
  }, [playingAudioMessageId, selectedLanguage, showToast, stopAudio]);

  return { playingAudioMessageId, handleAudio, stopAudio };
}

interface UseChatScrollOptions {
  messagesLength: number;
  streamingMessage: string;
}

interface UseChatScrollReturn {
  messagesEndRef: React.RefObject<HTMLDivElement>;
  messagesContainerRef: React.RefObject<HTMLDivElement>;
  showScrollButton: boolean;
  scrollToBottom: () => void;
}

/**
 * Custom hook for chat scroll behavior with debouncing
 */
export function useChatScroll({ messagesLength, streamingMessage }: UseChatScrollOptions): UseChatScrollReturn {
  const [showScrollButton, setShowScrollButton] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Handle scroll button visibility with debouncing
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    let timeoutId: ReturnType<typeof setTimeout>;
    const handleScroll = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        const { scrollTop, scrollHeight, clientHeight } = container;
        const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
        setShowScrollButton(!isNearBottom && messagesLength > 0);
      }, 50);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      container.removeEventListener('scroll', handleScroll);
      clearTimeout(timeoutId);
    };
  }, [messagesLength]);

  // Auto-scroll on new messages (only when near bottom)
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 200;

    if (isNearBottom) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messagesLength, streamingMessage]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  return {
    messagesEndRef,
    messagesContainerRef,
    showScrollButton,
    scrollToBottom
  };
}
