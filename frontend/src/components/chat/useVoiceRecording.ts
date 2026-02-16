/**
 * useVoiceRecording — Encapsulates MediaRecorder lifecycle + STT transcription.
 */
import { useState, useRef, useCallback } from 'react';
import { audio as audioApi } from '../../api';

interface UseVoiceRecordingOptions {
  selectedLanguage: string;
  isAuthenticated: boolean;
  onTranscription: (text: string) => void;
  onError?: (error: string) => void;
}

export function useVoiceRecording({
  selectedLanguage,
  isAuthenticated,
  onTranscription,
  onError,
}: UseVoiceRecordingOptions) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4',
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
          type: mediaRecorder.mimeType,
        });

        setIsTranscribing(true);
        try {
          const langCode = selectedLanguage === 'auto' ? 'auto' : selectedLanguage;
          const result = isAuthenticated
            ? await audioApi.speechToText(audioBlob, langCode)
            : await audioApi.speechToTextGuest(audioBlob, langCode);

          if (result.text) {
            onTranscription(result.text);
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
  }, [selectedLanguage, isAuthenticated, onTranscription, onError]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  }, []);

  return { isRecording, isTranscribing, startRecording, stopRecording };
}
