import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import type { TaskStatus } from '../types/api';

export default function TaskPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!taskId) return;

    const pollTask = async () => {
      try {
        const status = await api.getTaskStatus(taskId);
        setTaskStatus(status);

        if (status.state === 'SUCCESS' && status.result) {
          setTimeout(() => {
            navigate(`/content/${status.result.id}`);
          }, 2000);
        } else if (status.state === 'FAILURE') {
          setError(status.error || 'Task failed');
        }
      } catch (err: any) {
        setError(err.response?.data?.detail || err.message || 'Failed to get task status');
      }
    };

    const interval = setInterval(pollTask, 2000);
    pollTask();

    return () => clearInterval(interval);
  }, [taskId, navigate]);

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
        <div className="max-w-2xl mx-auto">
          <div className="glass-card p-8">
            <h1 className="text-2xl font-bold text-red-600 mb-4">Error</h1>
            <p className="text-gray-700">{error}</p>
            <button
              onClick={() => navigate('/upload')}
              className="mt-6 btn-primary"
            >
              Back to Upload
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="glass-card p-8">
          <h1 className="text-3xl font-bold mb-6 bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
            Processing Content
          </h1>

          {taskStatus && (
            <div className="space-y-6">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-gray-700">Status: {taskStatus.state}</span>
                  <span className="text-sm font-medium text-gray-700">{taskStatus.progress || 0}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-primary-600 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${taskStatus.progress || 0}%` }}
                  ></div>
                </div>
              </div>

              {taskStatus.stage && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-sm font-medium text-blue-900">Current Stage: {taskStatus.stage}</p>
                </div>
              )}

              {taskStatus.message && (
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <p className="text-sm text-gray-700">{taskStatus.message}</p>
                </div>
              )}

              {taskStatus.state === 'SUCCESS' && (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg text-green-700">
                  ✓ Processing complete! Redirecting...
                </div>
              )}
            </div>
          )}

          {!taskStatus && (
            <div className="flex justify-center items-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
