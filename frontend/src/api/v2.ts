/**
 * V2 API Extensions - OCR, Embeddings, STT, TTS
 */

import { API_BASE, getAuthHeader, handleResponse } from './client';

/**
 * OCR API - Extract text from images using GOT-OCR2
 */
export const ocr = {
  /**
   * Extract text from an image file
   * Uses GOT-OCR2 model with Tesseract fallback
   */
  async extractText(file: File | Blob, ocrType: 'document' | 'scene' | 'handwriting' = 'document'): Promise<{
    text: string;
    confidence?: number;
    language?: string;
    processing_time_ms: number;
    pages?: number;
    has_formulas?: boolean;
    formulas?: string[];
    has_tables?: boolean;
    tables?: Array<{ headers: string[]; rows: string[][] }>;
  }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/ocr/extract?ocr_type=${ocrType}`, {
      method: 'POST',
      headers: { ...getAuthHeader() },
      body: formData,
    });
    return handleResponse(response);
  },

  /**
   * Get OCR capabilities and supported formats
   */
  async getCapabilities(): Promise<{
    primary_model: string;
    fallback_model: string;
    supported_formats: string[];
    ocr_types: Array<{ type: string; description: string }>;
    languages: string[];
  }> {
    const response = await fetch(`${API_BASE}/ocr/capabilities`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },
};

/**
 * Embeddings API - Generate embeddings and rerank documents
 * Uses BGE-M3 for embeddings, BGE-Reranker-v2-M3 for reranking
 */
export const embeddings = {
  /**
   * Generate embeddings for texts
   * Useful for semantic search and similarity
   */
  async generate(texts: string[]): Promise<{
    embeddings: number[][];
    model: string;
    dimensions: number;
    count: number;
    processing_time_ms: number;
  }> {
    const response = await fetch(`${API_BASE}/embeddings/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ texts }),
    });
    return handleResponse(response);
  },

  /**
   * Rerank documents by relevance to a query
   * Returns documents sorted by relevance score
   */
  async rerank(query: string, documents: string[]): Promise<{
    results: Array<{ document: string; score: number; original_index: number }>;
    query: string;
    model: string;
    processing_time_ms: number;
  }> {
    const response = await fetch(`${API_BASE}/embeddings/rerank`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ query, documents }),
    });
    return handleResponse(response);
  },
};

/**
 * STT API - Speech-to-Text using Whisper V3 Turbo
 */
export const stt = {
  /**
   * Transcribe audio to text
   * Supports all Indian languages with auto-detection
   */
  async transcribe(audioFile: File | Blob, language = 'auto'): Promise<{
    text: string;
    language: string;
    confidence: number;
    segments?: Array<{ start: number; end: number; text: string }>;
    processing_time_ms: number;
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
   * Get supported STT languages
   */
  async getLanguages(): Promise<{
    languages: Array<{ code: string; name: string }>;
    model: string;
  }> {
    const response = await fetch(`${API_BASE}/stt/languages`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },
};

/**
 * TTS API - Text-to-Speech using MMS-TTS
 */
export const tts = {
  /**
   * Convert text to speech
   * Uses Edge TTS (online) or MMS-TTS (offline)
   */
  async synthesize(text: string, language = 'hi', voice = 'default'): Promise<{
    audio_id: string;
    audio_url: string;
    duration_ms: number;
    processing_time_ms: number;
  }> {
    const response = await fetch(`${API_BASE}/content/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ text, language, voice }),
    });
    return handleResponse(response);
  },

  /**
   * Get available TTS voices
   */
  async getVoices(): Promise<{
    voices: Array<{ id: string; name: string; languages: string[] }>;
  }> {
    const response = await fetch(`${API_BASE}/content/tts/voices`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },
};
