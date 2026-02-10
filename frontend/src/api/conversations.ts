/**
 * Conversations API - Manage chat conversations
 */

import { API_BASE, getAuthHeader, handleResponse } from './client';
import type { Conversation, Message } from './types';

export const conversations = {
  async list(): Promise<Conversation[]> {
    const response = await fetch(`${API_BASE}/chat/conversations`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse<Conversation[]>(response);
  },

  async create(title?: string, subject?: string, language?: string): Promise<Conversation> {
    const response = await fetch(`${API_BASE}/chat/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ title: title || 'New Conversation', subject: subject || 'General', language: language || 'en' }),
    });
    return handleResponse<Conversation>(response);
  },

  async get(id: string): Promise<Conversation> {
    const response = await fetch(`${API_BASE}/chat/conversations/${id}`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse<Conversation>(response);
  },

  async update(id: string, title: string): Promise<Conversation> {
    const response = await fetch(`${API_BASE}/chat/conversations/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...getAuthHeader() },
      body: JSON.stringify({ title }),
    });
    return handleResponse<Conversation>(response);
  },

  async delete(id: string): Promise<void> {
    const response = await fetch(`${API_BASE}/chat/conversations/${id}`, {
      method: 'DELETE',
      headers: { ...getAuthHeader() },
    });
    if (!response.ok) throw new Error('Failed to delete conversation');
  },

  async getMessages(id: string): Promise<Message[]> {
    const response = await fetch(`${API_BASE}/chat/conversations/${id}/messages`, {
      headers: { ...getAuthHeader() },
    });
    return handleResponse<Message[]>(response);
  },
};
