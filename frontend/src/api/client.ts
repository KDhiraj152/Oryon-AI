/**
 * API Client Core - Configuration and utilities
 *
 * Uses secure token management for XSS mitigation.
 * Supports request cancellation via AbortController.
 */

import {
  setAccessToken,
  setRefreshToken,
  getRefreshToken,
  clearTokens,
  getAuthHeader as secureGetAuthHeader
} from '../utils/secureTokens';

export const API_BASE = '/api/v2';

// Maximum retry attempts for failed requests
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;
const DEFAULT_TIMEOUT_MS = 30000;  // 30 second default timeout

/**
 * Map to track active requests for cancellation
 */
const activeRequests = new Map<string, AbortController>();

/**
 * Map to deduplicate in-flight requests with same key
 * Prevents redundant API calls when components mount/remount quickly
 */
const pendingRequests = new Map<string, Promise<Response>>();

/**
 * Generate cache key for request deduplication
 */
function getRequestKey(url: string, options?: RequestInit): string {
  const method = options?.method || 'GET';
  // Handle body as JSON string for proper deduplication key
  let bodyKey = '';
  if (options?.body) {
    bodyKey = typeof options.body === 'string'
      ? options.body.slice(0, 100)
      : JSON.stringify(options.body).slice(0, 100);
  }
  return `${method}:${url}:${bodyKey}`;
}

/**
 * Create a new AbortController with optional timeout
 * @param timeoutMs - Timeout in milliseconds (0 = no timeout)
 * @param requestId - Optional request ID for tracking
 */
export function createAbortController(
  timeoutMs = DEFAULT_TIMEOUT_MS,
  requestId?: string
): AbortController {
  const controller = new AbortController();

  // Track by request ID if provided
  if (requestId) {
    // Cancel any existing request with same ID
    cancelRequest(requestId);
    activeRequests.set(requestId, controller);
  }

  // Set timeout if specified
  if (timeoutMs > 0) {
    setTimeout(() => {
      if (!controller.signal.aborted) {
        controller.abort(new Error('Request timeout'));
      }
    }, timeoutMs);
  }

  return controller;
}

/**
 * Cancel an active request by ID
 */
export function cancelRequest(requestId: string): void {
  const controller = activeRequests.get(requestId);
  if (controller && !controller.signal.aborted) {
    controller.abort(new Error('Request cancelled'));
  }
  activeRequests.delete(requestId);
}

/**
 * Cancel all active requests
 */
export function cancelAllRequests(): void {
  for (const [, controller] of activeRequests) {
    if (!controller.signal.aborted) {
      controller.abort(new Error('All requests cancelled'));
    }
  }
  activeRequests.clear();
}

/**
 * Clean up completed request from tracking
 */
function cleanupRequest(requestId?: string): void {
  if (requestId) {
    activeRequests.delete(requestId);
  }
}

/**
 * Get authorization header with access token
 * Uses secure token manager for XSS mitigation
 */
export function getAuthHeader(): Record<string, string> {
  return secureGetAuthHeader();
}

/**
 * Sleep for a given number of milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Refresh the access token using the refresh token
 * Uses secure token storage
 */
export async function refreshToken(): Promise<boolean> {
  const refresh = getRefreshToken();
  if (!refresh) return false;

  try {
    const response = await fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refresh }),
    });

    if (response.ok) {
      const data = await response.json();
      setAccessToken(data.access_token);
      setRefreshToken(data.refresh_token);
      return true;
    }
  } catch {
    // Refresh failed
  }
  return false;
}

/**
 * Handle API response with automatic retry for network failures
 * @param response - Fetch response
 * @param retryFn - Optional function to retry the request
 * @param retryCount - Current retry count
 */
export async function handleResponse<T>(
  response: Response,
  retryFn?: () => Promise<Response>,
  retryCount = 0
): Promise<T> {
  if (!response.ok) {
    // Handle 401 - try to refresh token
    if (response.status === 401) {
      const refreshed = await refreshToken();
      if (refreshed && retryFn && retryCount < MAX_RETRIES) {
        await sleep(RETRY_DELAY_MS);
        const newResponse = await retryFn();
        return handleResponse(newResponse, retryFn, retryCount + 1);
      }
      clearTokens();
      globalThis.location.href = '/auth';
      throw new Error('Session expired');
    }

    // Handle 5xx errors with retry
    if (response.status >= 500 && retryFn && retryCount < MAX_RETRIES) {
      await sleep(RETRY_DELAY_MS * (retryCount + 1)); // Exponential backoff
      const newResponse = await retryFn();
      return handleResponse(newResponse, retryFn, retryCount + 1);
    }

    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || 'Request failed');
  }
  return response.json();
}

/**
 * Options for fetchWithRetry
 */
export interface FetchWithRetryOptions extends RequestInit {
  /** Request ID for cancellation tracking */
  requestId?: string;
  /** Timeout in milliseconds (default: 30000) */
  timeoutMs?: number;
  /** AbortController signal (will be merged with internal controller) */
  signal?: AbortSignal;
}

/**
 * Make a fetch request with automatic retry logic and abort support
 *
 * OPTIMIZATION: Deduplicates in-flight GET requests with same URL
 * to prevent redundant API calls from rapid component re-renders.
 */
export async function fetchWithRetry<T>(
  url: string,
  options: FetchWithRetryOptions = {}
): Promise<T> {
  const { requestId, timeoutMs = DEFAULT_TIMEOUT_MS, signal: externalSignal, ...fetchOptions } = options;

  // OPTIMIZATION: Deduplicate GET requests - return pending promise if exists
  const method = fetchOptions.method?.toUpperCase() || 'GET';
  const dedupeKey = method === 'GET' ? getRequestKey(url, options) : null;

  if (dedupeKey && pendingRequests.has(dedupeKey)) {
    // Return existing in-flight request
    try {
      const response = await pendingRequests.get(dedupeKey)!;
      return response.clone().json();
    } catch {
      // If cached request failed, continue with new request
    }
  }

  // Create abort controller for this request
  const controller = createAbortController(timeoutMs, requestId);

  // If external signal provided, abort when it does
  if (externalSignal) {
    if (externalSignal.aborted) {
      cleanupRequest(requestId);
      throw new Error('Request was cancelled');
    }
    externalSignal.addEventListener('abort', () => {
      controller.abort(externalSignal.reason || new Error('Request cancelled'));
    });
  }

  const makeRequest = () => fetch(url, {
    ...fetchOptions,
    signal: controller.signal
  });

  let lastError: Error | null = null;

  // Store promise for deduplication if this is a GET request
  const requestPromise = (async () => {
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const response = await makeRequest();
        const result = await handleResponse<T>(response, makeRequest, 0);
        cleanupRequest(requestId);
        return result;
      } catch (error) {
        lastError = error as Error;

        // Don't retry for abort/cancel errors
        if (controller.signal.aborted) {
          throw new Error(lastError.message || 'Request was cancelled');
        }

        // Don't retry for auth errors
        if (lastError.message === 'Session expired') {
          throw lastError;
        }

        // Wait before retrying
        if (attempt < MAX_RETRIES) {
          await sleep(RETRY_DELAY_MS * (attempt + 1));
        }
      }
    }
    throw lastError || new Error('Request failed after retries');
  })();

  // Track GET requests for deduplication
  if (dedupeKey) {
    pendingRequests.set(dedupeKey, requestPromise as unknown as Promise<Response>);
  }

  try {
    return await requestPromise;
  } finally {
    cleanupRequest(requestId);
    if (dedupeKey) {
      pendingRequests.delete(dedupeKey);
    }
  }
}
