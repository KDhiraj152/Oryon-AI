/**
 * Audio API - Text-to-Speech and Speech-to-Text
 */

import { API_BASE, getAuthHeader, handleResponse } from './client';

export const audio = {
  /**
   * Text-to-Speech (Authenticated)
   * Uses Edge TTS (primary) or MMS-TTS (fallback)
   */
  async textToSpeech(text: string, language = 'en', voice = 'default', speed = 1): Promise<{
    audio_url: string;
    duration?: number;
  }> {
    const response = await fetch(`${API_BASE}/audio/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ text, language, voice, speed }),
    });
    return handleResponse(response);
  },

  /**
   * Text-to-Speech (Guest - No Auth Required)
   * Returns base64 encoded audio data
   */
  async textToSpeechGuest(
    text: string,
    language = 'hi',
    gender: 'male' | 'female' = 'female',
    voice?: string,
    rate = '+0%',
    pitch = '+0Hz'
  ): Promise<{
    success: boolean;
    audio_data?: string;
    audio_format?: string;
    use_browser_tts?: boolean;
    error?: string;
  }> {
    const response = await fetch(`${API_BASE}/chat/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language, gender, voice, rate, pitch }),
    });
    return response.json();
  },

  /**
   * Get available TTS voices
   */
  async getVoices(): Promise<{
    voices: Array<{
      name: string;
      language: string;
      gender: string;
      locale: string;
    }>;
  }> {
    const response = await fetch(`${API_BASE}/chat/tts/voices`);
    return response.json();
  },

  /**
   * Speech-to-Text using Whisper Large V3 Turbo
   * Supports 99 languages including all Indian languages
   */
  async speechToText(audioFile: File | Blob, language = 'auto'): Promise<{
    text: string;
    language?: string;
    confidence?: number;
  }> {
    const formData = new FormData();
    formData.append('file', audioFile, 'recording.webm');

    const response = await fetch(`${API_BASE}/stt/transcribe?language=${language}`, {
      method: 'POST',
      headers: { ...getAuthHeader() },
      body: formData,
    });
    return handleResponse(response);
  },

  /**
   * Speech-to-Text (Guest - No Auth Required)
   * Uses Whisper Large V3 Turbo for transcription
   */
  async speechToTextGuest(audioBlob: Blob, language = 'auto'): Promise<{
    text: string;
    language?: string;
    success?: boolean;
    error?: string;
  }> {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    const response = await fetch(`${API_BASE}/stt/guest?language=${language}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Speech recognition failed');
    }
    return response.json();
  },

  /**
   * Play audio from base64 data
   */
  playBase64Audio(base64Data: string, format = 'audio/mpeg'): HTMLAudioElement {
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.codePointAt(i) ?? 0;
    }
    const audioBlob = new Blob([bytes], { type: format });
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.onended = () => URL.revokeObjectURL(audioUrl);
    audio.play();
    return audio;
  },
};
