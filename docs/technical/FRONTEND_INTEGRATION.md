# 🎨 Frontend Integration Guide

**Complete guide for integrating React/Vue/Angular frontend with ShikshaSetu API**

---

## 🚀 Quick Start

### Base Configuration

```javascript
// config.js
export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
  TIMEOUT: 30000, // 30 seconds
  MAX_FILE_SIZE: 100 * 1024 * 1024, // 100MB
  SUPPORTED_FILE_TYPES: ['application/pdf', 'image/png', 'image/jpeg'],
  POLL_INTERVAL: 3000 // 3 seconds for task polling
};

export const ENDPOINTS = {
  // Auth
  REGISTER: '/api/v1/auth/register',
  LOGIN: '/api/v1/auth/login',
  REFRESH: '/api/v1/auth/refresh',
  ME: '/api/v1/auth/me',
  
  // Content
  UPLOAD: '/api/v1/upload',
  UPLOAD_CHUNKED: '/api/v1/upload/chunked',
  PROCESS: '/api/v1/process',
  SIMPLIFY: '/api/v1/simplify',
  TRANSLATE: '/api/v1/translate',
  VALIDATE: '/api/v1/validate',
  TTS: '/api/v1/tts',
  
  // Tasks
  TASK_STATUS: '/api/v1/tasks',
  
  // Content Retrieval
  CONTENT: '/api/v1/content',
  AUDIO: '/api/v1/audio',
  FEEDBACK: '/api/v1/feedback',
  
  // Health
  HEALTH: '/health',
  HEALTH_DETAILED: '/health/detailed'
};
```

---

## 🔐 Authentication Implementation

### Auth Service (React)

```javascript
// services/authService.js
import axios from 'axios';
import { API_CONFIG, ENDPOINTS } from '../config';

class AuthService {
  constructor() {
    this.accessToken = localStorage.getItem('access_token');
    this.refreshToken = localStorage.getItem('refresh_token');
    this.setupInterceptors();
  }

  setupInterceptors() {
    // Add token to requests
    axios.interceptors.request.use(
      (config) => {
        if (this.accessToken) {
          config.headers.Authorization = `Bearer ${this.accessToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Handle token refresh
    axios.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            const newTokens = await this.refreshAccessToken();
            originalRequest.headers.Authorization = `Bearer ${newTokens.access_token}`;
            return axios(originalRequest);
          } catch (refreshError) {
            this.logout();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  async register(email, password, fullName) {
    const response = await axios.post(
      `${API_CONFIG.BASE_URL}${ENDPOINTS.REGISTER}`,
      { email, password, full_name: fullName }
    );

    this.setTokens(response.data);
    return response.data;
  }

  async login(email, password) {
    const response = await axios.post(
      `${API_CONFIG.BASE_URL}${ENDPOINTS.LOGIN}`,
      { email, password }
    );

    this.setTokens(response.data);
    return response.data;
  }

  async refreshAccessToken() {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await axios.post(
      `${API_CONFIG.BASE_URL}${ENDPOINTS.REFRESH}`,
      { refresh_token: this.refreshToken }
    );

    this.setTokens(response.data);
    return response.data;
  }

  async getCurrentUser() {
    const response = await axios.get(
      `${API_CONFIG.BASE_URL}${ENDPOINTS.ME}`
    );
    return response.data;
  }

  setTokens({ access_token, refresh_token }) {
    this.accessToken = access_token;
    this.refreshToken = refresh_token;
    localStorage.setItem('access_token', access_token);
    localStorage.setItem('refresh_token', refresh_token);
  }

  logout() {
    this.accessToken = null;
    this.refreshToken = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }

  isAuthenticated() {
    return !!this.accessToken;
  }
}

export default new AuthService();
```

### Auth Context (React)

```javascript
// context/AuthContext.jsx
import React, { createContext, useState, useEffect, useContext } from 'react';
import authService from '../services/authService';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    if (authService.isAuthenticated()) {
      try {
        const userData = await authService.getCurrentUser();
        setUser(userData);
      } catch (error) {
        authService.logout();
      }
    }
    setLoading(false);
  };

  const login = async (email, password) => {
    const tokens = await authService.login(email, password);
    const userData = await authService.getCurrentUser();
    setUser(userData);
    return tokens;
  };

  const register = async (email, password, fullName) => {
    const tokens = await authService.register(email, password, fullName);
    const userData = await authService.getCurrentUser();
    setUser(userData);
    return tokens;
  };

  const logout = () => {
    authService.logout();
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
```

---

## 📤 File Upload Implementation

### Upload Service with Progress

```javascript
// services/uploadService.js
import axios from 'axios';
import { API_CONFIG, ENDPOINTS } from '../config';

export const uploadFile = async (file, onProgress) => {
  // Validate file
  if (file.size > API_CONFIG.MAX_FILE_SIZE) {
    throw new Error(`File too large. Max size: ${API_CONFIG.MAX_FILE_SIZE / 1024 / 1024}MB`);
  }

  if (!API_CONFIG.SUPPORTED_FILE_TYPES.includes(file.type)) {
    throw new Error('Unsupported file type. Only PDF and images allowed.');
  }

  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(
    `${API_CONFIG.BASE_URL}${ENDPOINTS.UPLOAD}`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress?.(percentCompleted);
      }
    }
  );

  return response.data;
};

// Chunked upload for large files
export const uploadFileChunked = async (file, onProgress) => {
  const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB chunks
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  const fileId = `${Date.now()}_${file.name}`;

  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);

    const formData = new FormData();
    formData.append('file', chunk);
    formData.append('chunk_index', i);
    formData.append('total_chunks', totalChunks);
    formData.append('file_id', fileId);

    await axios.post(
      `${API_CONFIG.BASE_URL}${ENDPOINTS.UPLOAD_CHUNKED}`,
      formData
    );

    const percentCompleted = Math.round(((i + 1) * 100) / totalChunks);
    onProgress?.(percentCompleted);
  }

  return { file_id: fileId, message: 'Upload complete' };
};
```

### Upload Component (React)

```javascript
// components/FileUpload.jsx
import React, { useState } from 'react';
import { uploadFile } from '../services/uploadService';

export const FileUpload = ({ onUploadComplete }) => {
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      const result = await uploadFile(file, setProgress);
      onUploadComplete(result);
      setFile(null);
      setProgress(0);
    } catch (err) {
      setError(err.response?.data?.message || err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="file-upload">
      <input
        type="file"
        accept=".pdf,image/*"
        onChange={handleFileChange}
        disabled={uploading}
      />
      
      {file && (
        <div className="file-info">
          <p>{file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)</p>
          <button onClick={handleUpload} disabled={uploading}>
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      )}

      {uploading && (
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${progress}%` }}
          />
          <span>{progress}%</span>
        </div>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
};
```

---

## 🔄 Task Processing & Polling

### Process Service

```javascript
// services/processService.js
import axios from 'axios';
import { API_CONFIG, ENDPOINTS } from '../config';

export const processContent = async (processingParams) => {
  const response = await axios.post(
    `${API_CONFIG.BASE_URL}${ENDPOINTS.PROCESS}`,
    processingParams
  );
  return response.data;
};

export const getTaskStatus = async (taskId) => {
  const response = await axios.get(
    `${API_CONFIG.BASE_URL}${ENDPOINTS.TASK_STATUS}/${taskId}`
  );
  return response.data;
};

export const cancelTask = async (taskId, terminate = false) => {
  const response = await axios.delete(
    `${API_CONFIG.BASE_URL}${ENDPOINTS.TASK_STATUS}/${taskId}`,
    { params: { terminate } }
  );
  return response.data;
};

// Poll task until complete
export const pollTaskStatus = (taskId, onUpdate, onComplete, onError) => {
  const intervalId = setInterval(async () => {
    try {
      const status = await getTaskStatus(taskId);
      onUpdate(status);

      if (status.state === 'SUCCESS') {
        clearInterval(intervalId);
        onComplete(status.result);
      } else if (status.state === 'FAILURE') {
        clearInterval(intervalId);
        onError(new Error(status.error));
      }
    } catch (error) {
      clearInterval(intervalId);
      onError(error);
    }
  }, API_CONFIG.POLL_INTERVAL);

  return () => clearInterval(intervalId); // Cleanup function
};
```

### Processing Component (React)

```javascript
// components/ContentProcessor.jsx
import React, { useState, useEffect } from 'react';
import { processContent, pollTaskStatus, cancelTask } from '../services/processService';

export const ContentProcessor = ({ filePath }) => {
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleProcess = async () => {
    try {
      const task = await processContent({
        file_path: filePath,
        grade_level: 8,
        subject: 'Science',
        target_languages: ['Hindi', 'Tamil'],
        output_format: 'both',
        validation_threshold: 0.8
      });
      setTaskId(task.task_id);
    } catch (err) {
      setError(err.response?.data?.message || err.message);
    }
  };

  useEffect(() => {
    if (!taskId) return;

    const cleanup = pollTaskStatus(
      taskId,
      (status) => setStatus(status),
      (result) => {
        setResult(result);
        setStatus({ state: 'SUCCESS' });
      },
      (error) => setError(error.message)
    );

    return cleanup;
  }, [taskId]);

  const handleCancel = async () => {
    if (taskId) {
      await cancelTask(taskId);
      setTaskId(null);
      setStatus(null);
    }
  };

  return (
    <div className="content-processor">
      {!taskId && (
        <button onClick={handleProcess}>Start Processing</button>
      )}

      {status && (
        <div className="status">
          <h3>Status: {status.state}</h3>
          {status.stage && <p>Stage: {status.stage}</p>}
          {status.progress !== undefined && (
            <div className="progress-bar">
              <div style={{ width: `${status.progress}%` }} />
              <span>{status.progress}%</span>
            </div>
          )}
          {status.message && <p>{status.message}</p>}
          
          {status.state === 'PROCESSING' && (
            <button onClick={handleCancel}>Cancel</button>
          )}
        </div>
      )}

      {result && (
        <div className="result">
          <h3>Processing Complete!</h3>
          <div className="content">
            <h4>Simplified Text:</h4>
            <p>{result.simplified_text}</p>
            
            <h4>Translations:</h4>
            {Object.entries(result.translations).map(([lang, text]) => (
              <div key={lang}>
                <strong>{lang}:</strong> {text}
              </div>
            ))}
            
            <h4>Validation Score:</h4>
            <p>{(result.validation_score * 100).toFixed(1)}%</p>
            
            <h4>Audio:</h4>
            {Object.entries(result.audio_urls).map(([lang, url]) => (
              <audio key={lang} controls src={url}>
                {lang} Audio
              </audio>
            ))}
          </div>
        </div>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
};
```

---

## 🎵 Audio Player Implementation

```javascript
// components/AudioPlayer.jsx
import React, { useState, useRef } from 'react';
import axios from 'axios';
import { API_CONFIG, ENDPOINTS } from '../config';

export const AudioPlayer = ({ contentId, language }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const audioRef = useRef(null);

  const loadAudio = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(
        `${API_CONFIG.BASE_URL}${ENDPOINTS.AUDIO}/${contentId}`,
        {
          params: { language },
          responseType: 'blob'
        }
      );

      const audioBlob = new Blob([response.data], { type: 'audio/mpeg' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.load();
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to load audio');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="audio-player">
      <button onClick={loadAudio} disabled={loading}>
        {loading ? 'Loading...' : 'Load Audio'}
      </button>
      
      <audio ref={audioRef} controls>
        Your browser does not support audio playback.
      </audio>
      
      {error && <div className="error">{error}</div>}
    </div>
  );
};
```

---

## 🔔 Rate Limit Handling

```javascript
// utils/rateLimitHandler.js
export class RateLimitHandler {
  constructor() {
    this.retryQueue = [];
    this.processing = false;
  }

  async handleRequest(requestFn, maxRetries = 3) {
    let retries = 0;

    while (retries < maxRetries) {
      try {
        return await requestFn();
      } catch (error) {
        if (error.response?.status === 429) {
          const retryAfter = error.response.headers['retry-after'] || 60;
          console.log(`Rate limited. Retrying after ${retryAfter}s...`);
          
          await this.sleep(retryAfter * 1000);
          retries++;
        } else {
          throw error;
        }
      }
    }

    throw new Error('Max retries exceeded');
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export default new RateLimitHandler();
```

---

## ⚠️ Error Handling

```javascript
// utils/errorHandler.js
export const handleApiError = (error) => {
  if (error.response) {
    const { status, data } = error.response;

    switch (status) {
      case 400:
        return {
          title: 'Invalid Request',
          message: data.message || 'Please check your input and try again.'
        };
      
      case 401:
        return {
          title: 'Authentication Required',
          message: 'Please log in to continue.'
        };
      
      case 403:
        return {
          title: 'Access Denied',
          message: 'You don\'t have permission to access this resource.'
        };
      
      case 404:
        return {
          title: 'Not Found',
          message: 'The requested resource was not found.'
        };
      
      case 429:
        return {
          title: 'Rate Limit Exceeded',
          message: `Too many requests. Please try again in ${data.retry_after || 60} seconds.`
        };
      
      case 500:
        return {
          title: 'Server Error',
          message: 'Something went wrong on our end. Please try again later.'
        };
      
      case 503:
        return {
          title: 'Service Unavailable',
          message: 'The service is temporarily unavailable. Please try again later.'
        };
      
      default:
        return {
          title: 'Error',
          message: data.message || 'An unexpected error occurred.'
        };
    }
  } else if (error.request) {
    return {
      title: 'Network Error',
      message: 'Unable to connect to the server. Please check your internet connection.'
    };
  } else {
    return {
      title: 'Error',
      message: error.message || 'An unexpected error occurred.'
    };
  }
};
```

---

## 📊 Health Check Implementation

```javascript
// services/healthService.js
import axios from 'axios';
import { API_CONFIG, ENDPOINTS } from '../config';

export const checkHealth = async () => {
  const response = await axios.get(
    `${API_CONFIG.BASE_URL}${ENDPOINTS.HEALTH}`
  );
  return response.data;
};

export const checkDetailedHealth = async () => {
  const response = await axios.get(
    `${API_CONFIG.BASE_URL}${ENDPOINTS.HEALTH_DETAILED}`
  );
  return response.data;
};

// React Hook
export const useHealthCheck = (interval = 60000) => {
  const [health, setHealth] = React.useState(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    const checkStatus = async () => {
      try {
        const status = await checkHealth();
        setHealth(status);
      } catch (error) {
        setHealth({ status: 'unhealthy' });
      } finally {
        setLoading(false);
      }
    };

    checkStatus();
    const intervalId = setInterval(checkStatus, interval);

    return () => clearInterval(intervalId);
  }, [interval]);

  return { health, loading };
};
```

---

## 🎨 Complete Example Application (React)

```javascript
// App.jsx
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import ProcessContent from './pages/ProcessContent';

const PrivateRoute = ({ children }) => {
  const { user, loading } = useAuth();
  
  if (loading) return <div>Loading...</div>;
  return user ? children : <Navigate to="/login" />;
};

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route 
            path="/dashboard" 
            element={
              <PrivateRoute>
                <Dashboard />
              </PrivateRoute>
            } 
          />
          <Route 
            path="/process" 
            element={
              <PrivateRoute>
                <ProcessContent />
              </PrivateRoute>
            } 
          />
          <Route path="/" element={<Navigate to="/dashboard" />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
```

---

## 📱 Mobile Considerations

### Offline Support

```javascript
// utils/offlineStorage.js
export const saveToCache = (key, data) => {
  localStorage.setItem(key, JSON.stringify(data));
};

export const loadFromCache = (key) => {
  const data = localStorage.getItem(key);
  return data ? JSON.parse(data) : null;
};

export const cacheContent = (content) => {
  const cached = loadFromCache('cached_content') || [];
  cached.push(content);
  saveToCache('cached_content', cached);
};
```

### Network Status Detection

```javascript
// hooks/useNetworkStatus.js
import { useState, useEffect } from 'react';

export const useNetworkStatus = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
};
```

---

## 🧪 Testing

### API Service Tests (Jest)

```javascript
// services/__tests__/authService.test.js
import axios from 'axios';
import authService from '../authService';

jest.mock('axios');

describe('AuthService', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  test('login stores tokens', async () => {
    const mockResponse = {
      data: {
        access_token: 'test_access',
        refresh_token: 'test_refresh'
      }
    };
    axios.post.mockResolvedValue(mockResponse);

    await authService.login('test@example.com', 'password');

    expect(localStorage.getItem('access_token')).toBe('test_access');
    expect(localStorage.getItem('refresh_token')).toBe('test_refresh');
  });

  test('logout clears tokens', () => {
    localStorage.setItem('access_token', 'test');
    authService.logout();

    expect(localStorage.getItem('access_token')).toBeNull();
  });
});
```

---

## 🚀 Production Checklist

- [ ] **Environment Variables**: Configure `REACT_APP_API_URL` for production
- [ ] **Token Security**: Use secure storage (HttpOnly cookies recommended)
- [ ] **Error Boundaries**: Wrap components with error boundaries
- [ ] **Loading States**: Show proper loading indicators
- [ ] **Rate Limit UI**: Display retry timers to users
- [ ] **Offline Support**: Cache content for offline access
- [ ] **Analytics**: Track API errors and performance
- [ ] **Accessibility**: Ensure ARIA labels and keyboard navigation
- [ ] **Security**: Validate all user inputs before sending
- [ ] **Performance**: Implement debouncing for search/filter

---

## 📖 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Full API Reference**: [docs/API.md](API.md)
- **Backend README**: [../README.md](../README.md)
- **Sample Frontend**: [../frontend/](../frontend/)

---

**Last Updated:** 16 November 2025  
**Frontend Guide Version:** 1.0.0
