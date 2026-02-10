/**
 * AI Core API - Explainability, safety, prompts, and export
 */

import { API_BASE, getAuthHeader, handleResponse } from './client';
import type { ExplainabilityReport, SafetyCheckResult, ExportOptions } from './types';

export const aiCore = {
  /**
   * Get explainability report for a response
   */
  async explain(data: {
    query: string;
    response: string;
    sources?: Array<{ id: string; title: string; score: number }>;
    model_used?: string;
    latency_ms?: number;
  }): Promise<ExplainabilityReport> {
    const response = await fetch(`${API_BASE}/ai/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify(data),
    });
    return handleResponse<ExplainabilityReport>(response);
  },

  /**
   * Get cached explainability report
   */
  async getExplanation(requestId: string): Promise<ExplainabilityReport | null> {
    const response = await fetch(`${API_BASE}/ai/explain/${requestId}`, {
      headers: { ...getAuthHeader() },
    });
    if (response.status === 404) return null;
    return handleResponse<ExplainabilityReport>(response);
  },

  /**
   * Export conversation in various formats
   */
  async exportConversation(
    conversation: Array<{ role: string; content: string; timestamp?: string }>,
    options?: ExportOptions
  ): Promise<string> {
    const exportOptions = options ?? { format: 'markdown' };
    const response = await fetch(`${API_BASE}/ai/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({
        conversation,
        format: exportOptions.format,
        include_metadata: exportOptions.include_metadata ?? true,
        include_citations: exportOptions.include_citations ?? true,
      }),
    });
    const data = await handleResponse<{ content: string; format: string }>(response);
    return data.content;
  },

  /**
   * Get available export formats
   */
  async getExportFormats(): Promise<Array<{ id: string; name: string; extension: string }>> {
    const response = await fetch(`${API_BASE}/ai/export/formats`, {
      headers: { ...getAuthHeader() },
    });
    const data = await handleResponse<{ formats: Array<{ id: string; name: string; extension: string }> }>(response);
    return data.formats;
  },

  /**
   * Check content safety
   */
  async checkSafety(text: string): Promise<SafetyCheckResult> {
    const response = await fetch(`${API_BASE}/ai/safety/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ text }),
    });
    return handleResponse<SafetyCheckResult>(response);
  },

  /**
   * Redact secrets from text
   */
  async redactSecrets(text: string): Promise<{ redacted_text: string; secrets_found: number }> {
    const response = await fetch(`${API_BASE}/ai/safety/redact`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ text }),
    });
    return handleResponse(response);
  },

  /**
   * List available prompts
   */
  async listPrompts(): Promise<Array<{ name: string; description: string; version: string }>> {
    const response = await fetch(`${API_BASE}/ai/prompts`, {
      headers: { ...getAuthHeader() },
    });
    const data = await handleResponse<{ prompts: Array<{ name: string; description: string; version: string }> }>(response);
    return data.prompts;
  },

  /**
   * Render a prompt with variables
   */
  async renderPrompt(name: string, variables: Record<string, string>): Promise<string> {
    const response = await fetch(`${API_BASE}/ai/prompts/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ name, variables }),
    });
    const data = await handleResponse<{ rendered: string }>(response);
    return data.rendered;
  },

  /**
   * Get AI engine stats
   */
  async getStats(): Promise<{
    total_requests: number;
    average_latency_ms: number;
    cache_hit_rate: number;
    models_loaded: string[];
  }> {
    const response = await fetch(`${API_BASE}/ai/stats`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },

  /**
   * Health check for AI services
   */
  async healthCheck(): Promise<{ status: string; models: Record<string, boolean> }> {
    const response = await fetch(`${API_BASE}/ai/health`);
    return handleResponse(response);
  },
};

/**
 * Sandbox API - Execute code safely
 */
export const sandbox = {
  /**
   * Execute Python code safely
   */
  async executePython(code: string, timeout = 5): Promise<{
    success: boolean;
    output: string;
    error?: string;
    execution_time_ms: number;
  }> {
    const response = await fetch(`${API_BASE}/ai/sandbox/python`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ code, timeout }),
    });
    return handleResponse(response);
  },

  /**
   * Evaluate math expression
   */
  async calculate(expression: string): Promise<{
    success: boolean;
    result: number | string;
    error?: string;
  }> {
    const response = await fetch(`${API_BASE}/ai/sandbox/calculate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ expression }),
    });
    return handleResponse(response);
  },
};
