/**
 * Progress API - Learning progress and quiz management
 */

import { API_BASE, getAuthHeader, handleResponse } from './client';
import type { ProgressStats } from './types';

export const progress = {
  async getStats(): Promise<ProgressStats> {
    const response = await fetch(`${API_BASE}/progress/stats`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse<ProgressStats>(response);
  },

  async getSessions(limit = 10): Promise<{
    items: Array<{
      id: string;
      session_type?: string;
      topic?: string;
      title?: string;
      score?: number;
      duration_minutes?: number;
      created_at: string;
    }>;
  }> {
    const response = await fetch(`${API_BASE}/progress/sessions?limit=${limit}`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },

  async generateQuiz(topic: string, difficulty = 'medium', numQuestions = 10): Promise<{
    quiz_id: string;
    questions: Array<{
      id: string;
      question: string;
      options: string[];
      correct_answer: number;
      explanation?: string;
    }>;
  }> {
    const response = await fetch(`${API_BASE}/progress/quiz/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ topic, difficulty, num_questions: numQuestions }),
    });
    return handleResponse(response);
  },

  async submitQuiz(quizId: string, answers: Array<{ question_id: string; selected_answer: number }>): Promise<{
    score: number;
    total: number;
  }> {
    const response = await fetch(`${API_BASE}/progress/quiz/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ quiz_id: quizId, answers }),
    });
    return handleResponse(response);
  },
};
