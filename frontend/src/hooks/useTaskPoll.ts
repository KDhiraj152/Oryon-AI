import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';
import { api } from '../services/api';
import type { TaskStatus, TaskState } from '../types/api';

interface UseTaskPollOptions {
  taskId: string | null;
  interval?: number;
  enabled?: boolean;
  onSuccess?: (result: any) => void;
  onError?: (error: string) => void;
  onProgress?: (progress: number, stage: string) => void;
}

const TERMINAL_STATES = new Set<TaskState>(['SUCCESS', 'FAILURE', 'REVOKED']);

export function useTaskPoll(options: UseTaskPollOptions) {
  const {
    taskId,
    interval = 3000,
    enabled = true,
    onSuccess,
    onError,
    onProgress
  } = options;

  const queryClient = useQueryClient();
  const [isPolling, setIsPolling] = useState(false);
  const previousProgressRef = useRef<number>(0);
  const exponentialBackoffRef = useRef<number>(interval);

  const taskQuery = useQuery({
    queryKey: ['task', taskId],
    queryFn: () => {
      if (!taskId) throw new Error('No task ID provided');
      return api.getTaskStatus(taskId);
    },
    enabled: enabled && !!taskId && isPolling,
    refetchInterval: (query) => {
      const data = query.state.data;
      
      if (data && TERMINAL_STATES.has(data.state)) {
        setIsPolling(false);
        return false;
      }
      
      if (data && data.progress === previousProgressRef.current) {
        exponentialBackoffRef.current = Math.min(
          exponentialBackoffRef.current * 1.5,
          10000
        );
      } else {
        exponentialBackoffRef.current = interval;
      }
      
      return exponentialBackoffRef.current;
    },
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000)
  });

  useEffect(() => {
    if (enabled && taskId) {
      setIsPolling(true);
      exponentialBackoffRef.current = interval;
    }
  }, [enabled, taskId, interval]);

  useEffect(() => {
    if (!taskQuery.data) return;

    const status = taskQuery.data;

    if (status.progress !== previousProgressRef.current) {
      previousProgressRef.current = status.progress ?? 0;
      onProgress?.(status.progress ?? 0, status.stage ?? '');
    }

    if (status.state === 'SUCCESS') {
      onSuccess?.(status.result);
      setIsPolling(false);
    } else if (status.state === 'FAILURE') {
      onError?.(status.error || 'Task failed');
      setIsPolling(false);
    } else if (status.state === 'REVOKED') {
      onError?.('Task was cancelled');
      setIsPolling(false);
    }
  }, [taskQuery.data, onSuccess, onError, onProgress]);

  const stopPolling = () => {
    setIsPolling(false);
  };

  const startPolling = () => {
    if (taskId) {
      setIsPolling(true);
      exponentialBackoffRef.current = interval;
      queryClient.invalidateQueries({ queryKey: ['task', taskId] });
    }
  };

  const cancelTask = async (terminate: boolean = false) => {
    if (!taskId) return;
    
    try {
      await api.cancelTask(taskId, terminate);
      setIsPolling(false);
      queryClient.setQueryData(['task', taskId], (old: TaskStatus | undefined) => 
        old ? { ...old, state: 'REVOKED' as TaskState } : undefined
      );
    } catch (error) {
      console.error('Failed to cancel task:', error);
      throw error;
    }
  };

  return {
    taskStatus: taskQuery.data,
    state: taskQuery.data?.state,
    progress: taskQuery.data?.progress || 0,
    stage: taskQuery.data?.stage || '',
    message: taskQuery.data?.message || '',
    result: taskQuery.data?.result,
    error: taskQuery.data?.error,
    
    isLoading: taskQuery.isLoading,
    isError: taskQuery.isError,
    queryError: taskQuery.error,
    
    isPolling,
    startPolling,
    stopPolling,
    cancelTask,
    
    isPending: taskQuery.data?.state === 'PENDING',
    isProcessing: taskQuery.data?.state === 'PROCESSING' || taskQuery.data?.state === 'STARTED',
    isSuccess: taskQuery.data?.state === 'SUCCESS',
    isFailure: taskQuery.data?.state === 'FAILURE',
    isCancelled: taskQuery.data?.state === 'REVOKED'
  };
}
