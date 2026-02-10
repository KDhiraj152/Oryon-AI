/**
 * Content API - Upload, simplify, translate, and manage educational content
 */

import { API_BASE, getAuthHeader, handleResponse } from './client';
import type { TaskResponse, UploadResponse } from './types';

export const content = {
  async upload(
    file: File,
    processForQa = true
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('process_for_qa', processForQa.toString());

    const response = await fetch(`${API_BASE}/content/upload`, {
      method: 'POST',
      headers: { ...getAuthHeader() },
      body: formData,
    });
    return handleResponse<UploadResponse>(response);
  },

  async simplify(text: string): Promise<{
    simplified_text: string;
    task_id: string;
    status: string;
  }> {
    const response = await fetch(`${API_BASE}/content/simplify?wait=true`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ text }),
    });
    return handleResponse(response);
  },

  async translate(text: string, targetLanguages: string[]): Promise<{
    translated_text: string;
    translations: Record<string, string>;
    task_id: string;
    status: string;
  }> {
    const response = await fetch(`${API_BASE}/content/translate?wait=true`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ text, target_languages: targetLanguages }),
    });
    return handleResponse(response);
  },

  async validate(originalText: string, processedText: string): Promise<{
    is_valid: boolean;
    accuracy_score: number;
    issues: string[];
    task_id: string;
    status: string;
  }> {
    const response = await fetch(`${API_BASE}/content/validate?wait=true`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ original_text: originalText, processed_text: processedText }),
    });
    return handleResponse(response);
  },

  async getTask(taskId: string): Promise<TaskResponse> {
    const response = await fetch(`${API_BASE}/content/tasks/${taskId}`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse<TaskResponse>(response);
  },

  async get(contentId: string): Promise<{
    id: string;
    original_text: string;
    simplified_text?: string;
    translated_text?: string;
    language: string;
    subject: string;
    audio_url?: string;
    metadata: Record<string, unknown>;
  }> {
    const response = await fetch(`${API_BASE}/content/content/${contentId}`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },

  async getLibrary(params?: {
    language?: string;
    subject?: string;
    search?: string;
    page?: number;
    limit?: number;
  }): Promise<{
    items: Array<{
      id: string;
      title: string;
      subject: string;
      language: string;
      created_at: string;
    }>;
    total: number;
    page: number;
    pages: number;
  }> {
    const searchParams = new URLSearchParams();
    if (params?.language) searchParams.append('language', params.language);
    if (params?.subject) searchParams.append('subject', params.subject);
    if (params?.search) searchParams.append('search', params.search);
    if (params?.page) searchParams.append('page', params.page.toString());
    if (params?.limit) searchParams.append('limit', params.limit.toString());

    const response = await fetch(`${API_BASE}/content/library?${searchParams}`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },
};
