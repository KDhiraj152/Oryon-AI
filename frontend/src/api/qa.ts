/**
 * Q&A API - Document processing and question answering
 */

import { API_BASE, getAuthHeader, handleResponse } from './client';

export const qa = {
  async processDocument(contentId: string, chunkSize = 512, overlap = 50): Promise<{
    message: string;
    task_id: string;
    content_id: string;
    status_url: string;
  }> {
    const formData = new FormData();
    formData.append('content_id', contentId);
    formData.append('chunk_size', chunkSize.toString());
    formData.append('overlap', overlap.toString());

    const response = await fetch(`${API_BASE}/qa/process`, {
      method: 'POST',
      headers: { ...getAuthHeader() },
      body: formData,
    });
    return handleResponse(response);
  },

  async ask(contentId: string, question: string, topK = 3): Promise<{
    answer: string;
    sources: Array<{ chunk_id: string; text: string; score: number }>;
  }> {
    const formData = new FormData();
    formData.append('content_id', contentId);
    formData.append('question', question);
    formData.append('top_k', topK.toString());
    formData.append('wait', 'true');

    const response = await fetch(`${API_BASE}/qa/ask`, {
      method: 'POST',
      headers: { ...getAuthHeader() },
      body: formData,
    });
    return handleResponse(response);
  },
};

export const health = {
  async check(): Promise<{ status: string; database: string; timestamp: string }> {
    const response = await fetch('/health');
    return handleResponse(response);
  },
};
