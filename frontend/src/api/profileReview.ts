/**
 * Profile & Review API - Student profiles and teacher review
 */

import { API_BASE, getAuthHeader, handleResponse } from './client';
import type { StudentProfile, FlaggedResponse } from './types';

export const profile = {
  /**
   * Get current user's profile
   */
  async getProfile(): Promise<{
    profile: StudentProfile;
  }> {
    const response = await fetch(`${API_BASE}/profile/me`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },

  /**
   * Update user's profile
   */
  async updateProfile(updates: {
    language?: string;
  }): Promise<{
    success: boolean;
    profile: StudentProfile;
  }> {
    const response = await fetch(`${API_BASE}/profile/me`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify(updates),
    });
    return handleResponse(response);
  },
};

export const review = {
  /**
   * Get pending responses for teacher review
   */
  async getPending(limit = 20, offset = 0): Promise<{
    pending: FlaggedResponse[];
    total_pending: number;
    limit: number;
    offset: number;
  }> {
    const response = await fetch(`${API_BASE}/review/pending?limit=${limit}&offset=${offset}`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },

  /**
   * Get a specific flagged response
   */
  async getById(responseId: string): Promise<FlaggedResponse> {
    const response = await fetch(`${API_BASE}/review/${responseId}`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },

  /**
   * Submit a review for a flagged response
   */
  async submitReview(
    responseId: string,
    status: 'approved' | 'rejected' | 'improved',
    notes?: string,
    correctedResponse?: string
  ): Promise<{
    success: boolean;
    response: FlaggedResponse;
  }> {
    const response = await fetch(`${API_BASE}/review/${responseId}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ status, notes, corrected_response: correctedResponse }),
    });
    return handleResponse(response);
  },

  /**
   * Get review queue statistics
   */
  async getStats(): Promise<{
    total_flagged: number;
    pending_count: number;
    [key: string]: number;
  }> {
    const response = await fetch(`${API_BASE}/review/stats`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse(response);
  },
};
