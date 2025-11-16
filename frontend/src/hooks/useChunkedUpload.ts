import { useState, useCallback, useRef } from 'react';
import { api } from '../services/api';

interface UploadProgress {
  uploadId: string;
  fileName: string;
  totalChunks: number;
  uploadedChunks: number;
  progress: number;
  speed: number;
  timeRemaining: number;
  status: 'idle' | 'uploading' | 'paused' | 'completed' | 'error';
  error?: string;
}

interface UseChunkedUploadOptions {
  chunkSize?: number;
  maxRetries?: number;
  onComplete?: (uploadId: string) => void;
  onError?: (error: Error) => void;
}

export function useChunkedUpload(options: UseChunkedUploadOptions = {}) {
  const {
    chunkSize = 5 * 1024 * 1024,
    maxRetries = 3,
    onComplete,
    onError
  } = options;

  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const startTimeRef = useRef<number>(0);
  const uploadedBytesRef = useRef<number>(0);

  const generateUploadId = () => {
    return `${Date.now()}_${crypto.randomUUID()}`;
  };

  const calculateSpeed = (uploadedBytes: number, startTime: number): number => {
    const elapsedTime = (Date.now() - startTime) / 1000;
    return elapsedTime > 0 ? uploadedBytes / elapsedTime : 0;
  };

  const calculateTimeRemaining = (
    totalBytes: number,
    uploadedBytes: number,
    speed: number
  ): number => {
    const remainingBytes = totalBytes - uploadedBytes;
    return speed > 0 ? remainingBytes / speed : 0;
  };

  const uploadChunk = async (
    file: File,
    uploadId: string,
    chunkIndex: number,
    totalChunks: number,
    retries = 0
  ): Promise<void> => {
    const start = chunkIndex * chunkSize;
    const end = Math.min(start + chunkSize, file.size);
    const chunk = file.slice(start, end);

    try {
      await api.uploadChunk(
        chunk,
        file.name,
        uploadId,
        chunkIndex,
        totalChunks,
        undefined,
        (chunkProgress) => {
          const totalProgress = ((chunkIndex + chunkProgress / 100) / totalChunks) * 100;
          const uploadedBytes = uploadedBytesRef.current + (chunk.size * chunkProgress / 100);
          const speed = calculateSpeed(uploadedBytes, startTimeRef.current);
          const timeRemaining = calculateTimeRemaining(file.size, uploadedBytes, speed);

          setUploadProgress(prev => prev ? {
            ...prev,
            progress: totalProgress,
            uploadedChunks: chunkIndex,
            speed,
            timeRemaining
          } : null);
        }
      );

      uploadedBytesRef.current += chunk.size;
    } catch (error: any) {
      if (retries < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, retries)));
        return uploadChunk(file, uploadId, chunkIndex, totalChunks, retries + 1);
      }
      throw error;
    }
  };

  const startUpload = useCallback(async (file: File) => {
    const uploadId = generateUploadId();
    const totalChunks = Math.ceil(file.size / chunkSize);

    setUploadProgress({
      uploadId,
      fileName: file.name,
      totalChunks,
      uploadedChunks: 0,
      progress: 0,
      speed: 0,
      timeRemaining: 0,
      status: 'uploading'
    });

    setIsPaused(false);
    abortControllerRef.current = new AbortController();
    startTimeRef.current = Date.now();
    uploadedBytesRef.current = 0;

    try {
      for (let i = 0; i < totalChunks; i++) {
        if (isPaused) {
          setUploadProgress(prev => prev ? { ...prev, status: 'paused' } : null);
          return;
        }

        await uploadChunk(file, uploadId, i, totalChunks);

        setUploadProgress(prev => prev ? {
          ...prev,
          uploadedChunks: i + 1
        } : null);
      }

      setUploadProgress(prev => prev ? {
        ...prev,
        progress: 100,
        status: 'completed'
      } : null);

      onComplete?.(uploadId);
    } catch (error: any) {
      setUploadProgress(prev => prev ? {
        ...prev,
        status: 'error',
        error: error.message || 'Upload failed'
      } : null);
      onError?.(error);
    }
  }, [chunkSize, maxRetries, isPaused, onComplete, onError]);

  const pauseUpload = useCallback(() => {
    setIsPaused(true);
    abortControllerRef.current?.abort();
  }, []);

  const resumeUpload = useCallback((file: File) => {
    if (!uploadProgress || uploadProgress.status !== 'paused') return;

    setIsPaused(false);
    const resumeChunk = uploadProgress.uploadedChunks;
    const uploadId = uploadProgress.uploadId;
    const totalChunks = uploadProgress.totalChunks;

    (async () => {
      try {
        setUploadProgress(prev => prev ? { ...prev, status: 'uploading' } : null);

        for (let i = resumeChunk; i < totalChunks; i++) {
          if (isPaused) {
            setUploadProgress(prev => prev ? { ...prev, status: 'paused' } : null);
            return;
          }

          await uploadChunk(file, uploadId, i, totalChunks);

          setUploadProgress(prev => prev ? {
            ...prev,
            uploadedChunks: i + 1
          } : null);
        }

        setUploadProgress(prev => prev ? {
          ...prev,
          progress: 100,
          status: 'completed'
        } : null);

        onComplete?.(uploadId);
      } catch (error: any) {
        setUploadProgress(prev => prev ? {
          ...prev,
          status: 'error',
          error: error.message || 'Resume failed'
        } : null);
        onError?.(error);
      }
    })();
  }, [uploadProgress, isPaused, onComplete, onError]);

  const cancelUpload = useCallback(() => {
    abortControllerRef.current?.abort();
    setUploadProgress(null);
    setIsPaused(false);
  }, []);

  const resetUpload = useCallback(() => {
    setUploadProgress(null);
    setIsPaused(false);
    uploadedBytesRef.current = 0;
  }, []);

  return {
    uploadProgress,
    startUpload,
    pauseUpload,
    resumeUpload,
    cancelUpload,
    resetUpload,
    isUploading: uploadProgress?.status === 'uploading',
    isPaused: uploadProgress?.status === 'paused',
    isCompleted: uploadProgress?.status === 'completed',
    isError: uploadProgress?.status === 'error'
  };
}
