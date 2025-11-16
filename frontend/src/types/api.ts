// API Types for ShikshaSetu Backend Integration

export interface User {
  id: string;
  email: string;
  username: string;
  full_name?: string;
  organization?: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  full_name?: string;
  organization?: string;
}

export interface RefreshRequest {
  refresh_token: string;
}

export interface ProcessRequest {
  file_path: string;
  grade_level: number;
  subject: string;
  target_languages: string[];
  output_format?: 'text' | 'audio' | 'both';
  validation_threshold?: number;
}

export interface SimplifyRequest {
  text: string;
  grade_level: number;
  subject: string;
}

export interface TranslateRequest {
  text: string;
  target_languages: string[];
  subject?: string;
}

export interface ValidateRequest {
  original_text: string;
  processed_text: string;
  grade_level: number;
  subject: string;
  language: string;
}

export interface TTSRequest {
  text: string;
  language: string;
  subject?: string;
}

export interface ChunkedUploadRequest {
  file: File;
  upload_id: string;
  chunk_index: number;
  total_chunks: number;
  checksum?: string;
}

export type TaskState = 'PENDING' | 'STARTED' | 'PROCESSING' | 'SUCCESS' | 'FAILURE' | 'REVOKED';

export interface TaskStatus {
  task_id: string;
  state: TaskState;
  progress: number;
  stage: string;
  message?: string;
  result?: any;
  error?: string;
}

export interface ProcessedContent {
  id: string;
  original_text: string;
  simplified_text: string;
  translated_text?: string;  // For single content endpoint
  translations?: Record<string, string>;  // For library endpoint
  validation_score: number;
  audio_available?: boolean;  // For library endpoint
  audio_url?: string;  // For single content endpoint
  grade_level: number;
  subject: string;
  language: string;
  created_at?: string;
  updated_at?: string;
  metadata?: any;
}

export interface FeedbackRequest {
  content_id: string;
  rating: number;
  feedback_text?: string;
  issue_type?: string;
}

export interface HealthCheck {
  status: string;
  timestamp: string;
}

export interface DetailedHealthCheck {
  status: string;
  checks: {
    database: {
      status: string;
      latency_ms: number;
    };
    redis: {
      status: string;
      latency_ms: number;
    };
    celery: {
      active_workers: number;
      active_tasks: number;
    };
    storage: {
      disk_usage_percent: number;
      free_gb: number;
    };
    system: {
      cpu: Record<string, unknown>;
      memory: Record<string, unknown>;
    };
  };
}

export interface ApiError {
  error: string;
  message: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface LibraryFilters {
  language?: string;
  grade?: number;
  subject?: string;
  limit?: number;
  offset?: number;
}

export interface SearchParams {
  q: string;
  limit?: number;
}

export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset: number;
  retryAfter?: number;
}

export const LANGUAGES = [
  'Hindi',
  'Tamil',
  'Telugu',
  'Bengali',
  'Marathi',
  'Gujarati',
  'Kannada',
  'Malayalam',
  'Punjabi',
  'Odia'
] as const;

export const SUBJECTS = [
  'Science',
  'Mathematics',
  'Social Studies',
  'English',
  'Hindi',
  'Computer Science'
] as const;

export const GRADE_LEVELS = [5, 6, 7, 8, 9, 10, 11, 12] as const;

export type Language = typeof LANGUAGES[number];
export type Subject = typeof SUBJECTS[number];
export type GradeLevel = typeof GRADE_LEVELS[number];
