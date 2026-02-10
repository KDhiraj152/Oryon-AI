/**
 * API Index - Modular API client for Shiksha Setu
 *
 * This file re-exports all API modules for backward compatibility.
 * Individual modules can be imported directly for tree-shaking benefits.
 *
 * @example
 * // Import everything (legacy)
 * import api from './api';
 * api.auth.login(email, password);
 *
 * // Import specific modules (recommended)
 * import { auth } from './api/auth';
 * import { chat } from './api/chat';
 */

// Re-export all types
export * from './types';

// Re-export client utilities (for custom usage)
export { API_BASE, fetchWithRetry, getAuthHeader, handleResponse } from './client';

// Re-export auth module
export { auth, refreshToken } from './auth';

// Re-export conversation module
export { conversations } from './conversations';

// Re-export chat module
export { chat } from './chat';
export type { SendMessageOptions } from './chat';

// Re-export content module
export { content } from './content';

// Re-export audio module
export { audio } from './audio';

// Re-export progress module
export { progress } from './progress';

// Re-export Q&A and health modules
export { qa, health } from './qa';

// Re-export AI Core and sandbox modules
export { aiCore, sandbox } from './aiCore';

// Re-export V2 API extensions
export { ocr, embeddings, stt, tts } from './v2';

// Re-export profile and review modules
export { profile, review } from './profileReview';

// Re-export system API
export { system } from './system';
export type {
  HardwareStatus,
  ModelsStatus,
  DeviceCapabilities,
  ModelInfo,
  CacheStatus,
  SystemHealth,
  BatchMetrics
} from './system';

// Import all modules for default export
import { auth } from './auth';
import { conversations } from './conversations';
import { chat } from './chat';
import { content } from './content';
import { audio } from './audio';
import { progress } from './progress';
import { qa, health } from './qa';
import { aiCore, sandbox } from './aiCore';
import { ocr, embeddings, stt, tts } from './v2';
import { profile, review } from './profileReview';

// Default export with all API modules (backward compatible)
export default {
  auth,
  conversations,
  chat,
  content,
  audio,
  progress,
  qa,
  health,
  aiCore,
  sandbox,
  // V2 API additions
  ocr,
  embeddings,
  stt,
  tts,
  profile,
  review,
};
