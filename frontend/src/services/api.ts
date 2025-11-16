import axios, { type AxiosInstance, AxiosError, type InternalAxiosRequestConfig } from 'axios';
import type {
  User,
  TokenResponse,
  LoginRequest,
  RegisterRequest,
  RefreshRequest,
  ProcessRequest,
  SimplifyRequest,
  TranslateRequest,
  ValidateRequest,
  TTSRequest,
  TaskStatus,
  ProcessedContent,
  FeedbackRequest,
  HealthCheck,
  DetailedHealthCheck,
  RateLimitInfo,
  ApiError,
  PaginatedResponse,
  LibraryFilters,
  SearchParams
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

let isRefreshing = false;
let refreshSubscribers: ((token: string) => void)[] = [];

const subscribeTokenRefresh = (callback: (token: string) => void) => {
  refreshSubscribers.push(callback);
};

const onTokenRefreshed = (token: string) => {
  for (const callback of refreshSubscribers) {
    callback(token);
  }
  refreshSubscribers = [];
};

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('access_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => {
    const rateLimitInfo: RateLimitInfo = {
      limit: Number.parseInt(response.headers['x-ratelimit-limit'] || '0'),
      remaining: Number.parseInt(response.headers['x-ratelimit-remaining'] || '0'),
      reset: Number.parseInt(response.headers['x-ratelimit-reset'] || '0'),
      retryAfter: response.headers['retry-after'] 
        ? Number.parseInt(response.headers['retry-after']) 
        : undefined
    };
    
    (response as any).rateLimitInfo = rateLimitInfo;
    return response;
  },
  async (error: AxiosError<ApiError>) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise((resolve) => {
          subscribeTokenRefresh((token: string) => {
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${token}`;
            }
            resolve(apiClient(originalRequest));
          });
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const refreshToken = localStorage.getItem('refresh_token');
        if (!refreshToken) throw new Error('No refresh token');

        const response = await axios.post<TokenResponse>(
          `${API_BASE_URL}/api/v1/auth/refresh`,
          { refresh_token: refreshToken }
        );

        const { access_token, refresh_token } = response.data;
        localStorage.setItem('access_token', access_token);
        localStorage.setItem('refresh_token', refresh_token);

        onTokenRefreshed(access_token);
        isRefreshing = false;

        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
        }
        return apiClient(originalRequest);
      } catch (refreshError) {
        isRefreshing = false;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        globalThis.location.href = '/login';
        throw refreshError;
      }
    }

    if (error.response?.status === 429) {
      const retryAfter = error.response.headers['retry-after'];
      const delay = retryAfter ? Number.parseInt(retryAfter) * 1000 : 5000;
      await new Promise(resolve => setTimeout(resolve, delay));
      return apiClient(originalRequest);
    }

    throw error;
  }
);

class ApiService {
  async register(data: RegisterRequest): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>('/api/v1/auth/register', data);
    return response.data;
  }

  async login(data: LoginRequest): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>('/api/v1/auth/login', data);
    return response.data;
  }

  async refreshToken(data: RefreshRequest): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>('/api/v1/auth/refresh', data);
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await apiClient.get<User>('/api/v1/auth/me');
    return response.data;
  }

  async uploadFile(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ file_path: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<{ file_path: string }>(
      '/api/v1/upload',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100;
            onProgress?.(progress);
          }
        }
      }
    );
    return response.data;
  }

  async uploadChunk(
    chunk: Blob,
    fileName: string,
    uploadId: string,
    chunkIndex: number,
    totalChunks: number,
    checksum?: string,
    onProgress?: (progress: number) => void
  ): Promise<{ status: string; message: string }> {
    const formData = new FormData();
    formData.append('file', chunk, fileName);

    const metadata: Record<string, string | number> = {
      filename: fileName,
      upload_id: uploadId,
      chunk_index: chunkIndex,
      total_chunks: totalChunks
    };

    if (checksum) {
      metadata.checksum = checksum;
    }

    formData.append('metadata', JSON.stringify(metadata));

    const response = await apiClient.post<{ status: string; message: string }>(
      '/api/v1/upload/chunked',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100;
            onProgress?.(progress);
          }
        }
      }
    );
    return response.data;
  }

  async processContent(data: ProcessRequest): Promise<{ task_id: string; status: string; message?: string }> {
    const response = await apiClient.post<{ task_id: string; status: string; message?: string }>(
      '/api/v1/process',
      {
        grade_level: data.grade_level,
        subject: data.subject,
        target_languages: data.target_languages,
        output_format: data.output_format || 'both',
        validation_threshold: data.validation_threshold || 0.8
      },
      {
        params: {
          file_path: data.file_path
        }
      }
    );
    return response.data;
  }

  async simplifyText(data: SimplifyRequest): Promise<{ task_id: string; status: string }> {
    const response = await apiClient.post<{ task_id: string; status: string }>(
      '/api/v1/simplify',
      data
    );
    return response.data;
  }

  async translateText(data: TranslateRequest): Promise<{ task_id: string; status: string }> {
    const response = await apiClient.post<{ task_id: string; status: string }>(
      '/api/v1/translate',
      data
    );
    return response.data;
  }

  async validateContent(data: ValidateRequest): Promise<{ task_id: string; status: string }> {
    const response = await apiClient.post<{ task_id: string; status: string }>(
      '/api/v1/validate',
      data
    );
    return response.data;
  }

  async generateAudio(data: TTSRequest): Promise<{ task_id: string; status: string }> {
    const response = await apiClient.post<{ task_id: string; status: string }>(
      '/api/v1/tts',
      data
    );
    return response.data;
  }

  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    const response = await apiClient.get<TaskStatus>(`/api/v1/tasks/${taskId}`);
    return response.data;
  }

  async cancelTask(taskId: string, terminate: boolean = false): Promise<{ message: string }> {
    const response = await apiClient.delete<{ message: string }>(
      `/api/v1/tasks/${taskId}`,
      { params: { terminate } }
    );
    return response.data;
  }

  async getContent(contentId: string): Promise<ProcessedContent> {
    const response = await apiClient.get<ProcessedContent>(`/api/v1/content/${contentId}`);
    return response.data;
  }

  getAudioUrl(contentId: string, language?: string): string {
    const params = language ? `?language=${language}` : '';
    return `${API_BASE_URL}/api/v1/audio/${contentId}${params}`;
  }

  async submitFeedback(data: FeedbackRequest): Promise<{ message: string }> {
    const response = await apiClient.post<{ message: string }>('/api/v1/feedback', data);
    return response.data;
  }

  async getLibrary(filters: LibraryFilters): Promise<PaginatedResponse<ProcessedContent>> {
    const response = await apiClient.get<PaginatedResponse<ProcessedContent>>(
      '/api/v1/library',
      { params: filters }
    );
    return response.data;
  }

  async searchContent(params: SearchParams): Promise<{ results: ProcessedContent[] }> {
    const response = await apiClient.get<{ results: ProcessedContent[] }>(
      '/api/v1/content/search',
      { params }
    );
    return response.data;
  }

  async getHealth(): Promise<HealthCheck> {
    const response = await apiClient.get<HealthCheck>('/health');
    return response.data;
  }

  async getDetailedHealth(): Promise<DetailedHealthCheck> {
    const response = await apiClient.get<DetailedHealthCheck>('/health/detailed');
    return response.data;
  }
}

export const api = new ApiService();
export default api;
