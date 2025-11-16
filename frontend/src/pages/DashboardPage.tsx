import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { api } from '../services/api';
import type { ProcessedContent } from '../types/api';

export default function DashboardPage() {
  const { user } = useAuthStore();
  const [recentContent, setRecentContent] = useState<ProcessedContent[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchRecentContent = async () => {
      try {
        const response = await api.getLibrary({ limit: 5, offset: 0 });
        setRecentContent(response.items);
      } catch (err) {
        console.error('Failed to load recent content:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchRecentContent();
  }, []);

  const renderRecentContent = () => {
    if (isLoading) {
      return (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      );
    }

    if (recentContent.length === 0) {
      return (
        <div className="text-center py-12">
          <p className="text-gray-600 mb-4">No content yet</p>
          <Link to="/upload" className="btn-primary">
            Upload Your First Content
          </Link>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {recentContent.map((item) => (
          <Link
            key={item.id}
            to={`/content/${item.id}`}
            className="block p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 mb-1">{item.subject}</h3>
                <p className="text-sm text-gray-600 line-clamp-2 mb-2">
                  {item.simplified_text.substring(0, 150)}...
                </p>
                <div className="flex gap-3 text-xs text-gray-500">
                  <span>Grade {item.grade_level}</span>
                  <span>•</span>
                  <span>{item.language}</span>
                  {item.created_at && (
                    <>
                      <span>•</span>
                      <span>{new Date(item.created_at).toLocaleDateString()}</span>
                    </>
                  )}
                </div>
              </div>
              {item.audio_available && (
                <div className="ml-4 text-green-600">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                  </svg>
                </div>
              )}
            </div>
          </Link>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
              Welcome back, {user?.full_name || user?.username || 'Educator'}!
            </h1>
            <p className="text-gray-600">Here is the latest summary of your processed content.</p>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Total Processed</h3>
            <p className="text-2xl font-bold text-gray-900">{recentContent.length}</p>
          </div>
        </div>

        <div className="glass-card p-8">
          <h2 className="text-2xl font-semibold mb-6">Recent Content</h2>
          {renderRecentContent()}
        </div>
      </div>
    </div>
  );
}
