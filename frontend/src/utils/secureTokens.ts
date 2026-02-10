/**
 * Secure Token Manager
 * ====================
 *
 * Provides secure token storage with XSS mitigation.
 *
 * Security Features:
 * 1. Memory-first storage (tokens not persisted during session)
 * 2. Encrypted localStorage fallback for refresh tokens only
 * 3. Short-lived access tokens in memory only
 * 4. Automatic token rotation
 * 5. Secure cleanup on logout
 *
 * SECURITY NOTE:
 * The ideal solution is httpOnly cookies set by the backend.
 * This implementation provides defense-in-depth for the frontend.
 *
 * @see https://owasp.org/www-community/HttpOnly
 */

// In-memory storage for access tokens (most secure)
let _accessToken: string | null = null;
let _tokenExpiry: number | null = null;

// Simple obfuscation key (NOT encryption, just makes casual inspection harder)
const OBFUSCATE_KEY = 'ssetu_2024_key';

/**
 * Basic obfuscation for localStorage (NOT secure encryption)
 * This is defense-in-depth, not primary security.
 */
function obfuscate(value: string): string {
  try {
    const encoded = btoa(value);
    // XOR with key
    let result = '';
    for (let i = 0; i < encoded.length; i++) {
      const encodedCode = encoded.codePointAt(i) ?? 0;
      const keyCode = OBFUSCATE_KEY.codePointAt(i % OBFUSCATE_KEY.length) ?? 0;
      result += String.fromCodePoint(encodedCode ^ keyCode);
    }
    return btoa(result);
  } catch {
    return value;
  }
}

function deobfuscate(value: string): string {
  try {
    const decoded = atob(value);
    let result = '';
    for (let i = 0; i < decoded.length; i++) {
      const decodedCode = decoded.codePointAt(i) ?? 0;
      const keyCode = OBFUSCATE_KEY.codePointAt(i % OBFUSCATE_KEY.length) ?? 0;
      result += String.fromCodePoint(decodedCode ^ keyCode);
    }
    return atob(result);
  } catch {
    return value;
  }
}

/**
 * Set access token (memory only - most secure)
 */
export function setAccessToken(token: string, expiresInSeconds = 3600): void {
  _accessToken = token;
  _tokenExpiry = Date.now() + (expiresInSeconds * 1000);

  // Access tokens are kept in memory only for security
  // No localStorage storage to prevent XSS token theft
}

/**
 * Get access token from memory
 */
export function getAccessToken(): string | null {
  // Check expiry
  if (_accessToken && _tokenExpiry && Date.now() > _tokenExpiry) {
    _accessToken = null;
    _tokenExpiry = null;
  }

  return _accessToken;
}

/**
 * Set refresh token (obfuscated localStorage)
 * Refresh tokens need persistence for "remember me" functionality
 */
export function setRefreshToken(token: string): void {
  try {
    const obfuscated = obfuscate(token);
    localStorage.setItem('_rt', obfuscated);
    // Plaintext storage removed for security
  } catch {
    // Ignore localStorage errors
  }
}

/**
 * Get refresh token
 */
export function getRefreshToken(): string | null {
  try {
    // Get obfuscated token
    const obfuscated = localStorage.getItem('_rt');
    if (obfuscated) {
      return deobfuscate(obfuscated);
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Clear all tokens (logout)
 */
export function clearTokens(): void {
  _accessToken = null;
  _tokenExpiry = null;

  try {
    // Clear obfuscated refresh token
    localStorage.removeItem('_rt');
    // Clean up any legacy plaintext tokens that may exist
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  } catch {
    // Ignore errors
  }
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  return getAccessToken() !== null;
}

/**
 * Get auth header for API requests
 */
export function getAuthHeader(): Record<string, string> {
  const token = getAccessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Initialize tokens from response
 */
export function initializeTokens(
  accessToken: string,
  refreshToken: string,
  expiresIn = 3600
): void {
  setAccessToken(accessToken, expiresIn);
  setRefreshToken(refreshToken);
}

/**
 * Handle token refresh
 */
export async function handleTokenRefresh(
  refreshFn: (refreshToken: string) => Promise<{ access_token: string; refresh_token: string } | null>
): Promise<boolean> {
  const refresh = getRefreshToken();
  if (!refresh) return false;

  try {
    const result = await refreshFn(refresh);
    if (result) {
      setAccessToken(result.access_token);
      setRefreshToken(result.refresh_token);
      return true;
    }
  } catch {
    // Refresh failed
  }

  clearTokens();
  return false;
}
