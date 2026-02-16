import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock fetch globally
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

describe('Audio API', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('should export audio module with expected methods', async () => {
    const { audio } = await import('../api/audio');
    expect(audio).toBeDefined();
    expect(typeof audio.textToSpeech).toBe('function');
    expect(typeof audio.speechToText).toBe('function');
    expect(typeof audio.textToSpeechGuest).toBe('function');
    expect(typeof audio.speechToTextGuest).toBe('function');
    expect(typeof audio.getVoices).toBe('function');
    expect(typeof audio.playBase64Audio).toBe('function');
  });

  it('should export v2 stt/tts modules', async () => {
    const { stt, tts, ocr, embeddings } = await import('../api/v2');
    expect(stt).toBeDefined();
    expect(typeof stt.transcribe).toBe('function');
    expect(typeof stt.getLanguages).toBe('function');
    expect(tts).toBeDefined();
    expect(typeof tts.synthesize).toBe('function');
    expect(typeof tts.getVoices).toBe('function');
    expect(ocr).toBeDefined();
    expect(embeddings).toBeDefined();
  });

  it('should call correct endpoint for speechToText', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ text: 'hello', language: 'en' }),
    });

    const { audio } = await import('../api/audio');
    const blob = new Blob(['audio'], { type: 'audio/webm' });
    await audio.speechToText(blob as unknown as File, 'en');

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const call = mockFetch.mock.calls[0];
    expect(call[0]).toContain('/stt/transcribe');
    expect(call[0]).toContain('language=en');
  });
});
