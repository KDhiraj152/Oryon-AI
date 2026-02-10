/**
 * API Types - Shared type definitions for API client
 */

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role?: string;
  preferences?: Partial<UserPreferences>;
  created_at?: string;
}

export interface UserPreferences {
  default_target_language: string;
  ui_language: string;
  theme: string;
  auto_detect_source: boolean;
  include_audio: boolean;
  voice_type: string;
  speech_speed: number;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: User;
}

export interface Conversation {
  id: string;
  title: string;
  language: string;
  subject?: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message_preview?: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  attachments?: Attachment[];
  created_at: string;
  status: string;
  metadata?: Record<string, unknown>;
}

export interface Attachment {
  id: string;
  type: string;
  filename: string;
  url: string;
  size: number;
  mime_type: string;
}

export interface ChatContext {
  target_language?: string;
  source_language?: string;
  task_hint?: string;
}

export interface UploadResponse {
  status: string;
  content_id: string;
  file_id: string;
  file_path: string;
  filename: string;
  size: number;
  extracted_text: string;
  qa_processing?: {
    enabled: boolean;
    task_id?: string;
    status_url?: string;
  };
}

export interface TaskResponse {
  task_id: string;
  state: string;
  message?: string;
  result?: unknown;
}

export interface ProgressStats {
  total_sessions: number;
  total_time_minutes: number;
  quizzes_completed: number;
  average_score: number;
  current_streak: number;
  longest_streak: number;
  topics_explored: number;
  content_consumed: number;
}

export interface ExplainabilityReport {
  request_id: string;
  query: string;
  sources: Array<{
    id: string;
    title: string;
    score: number;
    excerpt?: string;
  }>;
  model_used: string;
  latency_ms: number;
  token_usage: {
    input: number;
    output: number;
    total: number;
  };
  reasoning_steps?: string[];
}

export interface SafetyCheckResult {
  is_safe: boolean;
  issues: string[];
  redacted_text?: string;
}

export interface ExportOptions {
  format: 'json' | 'markdown' | 'text';
  include_metadata?: boolean;
  include_citations?: boolean;
}

export interface StudentProfile {
  language: string;
}

export interface FlaggedResponse {
  id: string;
  conversation_id: string;
  user_id: string | null;
  user_query: string;
  ai_response: string;
  reason: 'low_confidence' | 'no_sources' | 'safety_concern' | 'user_reported';
  confidence: number;
  sources_count: number;
  created_at: string;
  status: 'pending' | 'approved' | 'rejected' | 'improved';
  reviewer_id: string | null;
  reviewed_at: string | null;
  reviewer_notes: string | null;
  corrected_response: string | null;
}
