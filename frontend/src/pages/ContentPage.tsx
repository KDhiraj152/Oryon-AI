import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../services/api';
import type { ProcessedContent } from '../types/api';

export default function ContentPage() {
  const { contentId } = useParams<{ contentId: string }>();
  const [content, setContent] = useState<ProcessedContent | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'simplified' | 'translations'>('simplified');

  useEffect(() => {
    if (!contentId) return;

    const fetchContent = async () => {
      try {
        const data = await api.getContent(contentId);
        setContent(data);
      } catch (err: any) {
        setError(err.response?.data?.detail || err.message || 'Failed to load content');
      } finally {
        setIsLoading(false);
      }
    };

    fetchContent();
  }, [contentId]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error || !content) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="glass-card p-8">
            <h1 className="text-2xl font-bold text-red-600 mb-4">Error</h1>
            <p className="text-gray-700">{error || 'Content not found'}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
            {content.subject}
          </h1>
          <div className="flex gap-4 text-sm text-gray-600">
            <span>Grade {content.grade_level}</span>
            <span>•</span>
            <span>Language: {content.language}</span>
            <span>•</span>
            <span>Score: {(content.validation_score * 100).toFixed(0)}%</span>
          </div>
        </div>

        <div className="glass-card p-8 mb-6">
          <h2 className="text-xl font-semibold mb-4">Original Text</h2>
          <div className="prose max-w-none">
            <p className="text-gray-700 whitespace-pre-wrap">{content.original_text}</p>
          </div>
        </div>

        <div className="glass-card p-8 mb-6">
          <div className="flex gap-4 mb-6 border-b border-gray-200">
            <button
              onClick={() => setActiveTab('simplified')}
              className={`pb-3 px-4 font-medium transition-colors ${
                activeTab === 'simplified'
                  ? 'text-primary-600 border-b-2 border-primary-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Simplified Text
            </button>
            <button
              onClick={() => setActiveTab('translations')}
              className={`pb-3 px-4 font-medium transition-colors ${
                activeTab === 'translations'
                  ? 'text-primary-600 border-b-2 border-primary-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Translations
            </button>
          </div>

          {activeTab === 'simplified' && (
            <div className="prose max-w-none">
              <p className="text-gray-700 whitespace-pre-wrap">{content.simplified_text}</p>
            </div>
          )}

          {activeTab === 'translations' && (
            <div className="space-y-4">
              {content.translations && Object.entries(content.translations).map(([lang, text]) => (
                <div key={lang} className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-semibold text-gray-900 mb-2">{lang}</h3>
                  <p className="text-gray-700 whitespace-pre-wrap">{text}</p>
                </div>
              ))}
              {content.translated_text && !content.translations && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-semibold text-gray-900 mb-2">{content.language}</h3>
                  <p className="text-gray-700 whitespace-pre-wrap">{content.translated_text}</p>
                </div>
              )}
              {!content.translations && !content.translated_text && (
                <p className="text-gray-500">No translations available</p>
              )}
            </div>
          )}
        </div>

        {(content.audio_available || content.audio_url) && (
          <div className="glass-card p-8">
            <h2 className="text-xl font-semibold mb-4">Audio</h2>
            <audio controls className="w-full">
              <source src={content.audio_url || api.getAudioUrl(content.id)} type="audio/mpeg" />
              <track kind="captions" />
              Your browser does not support the audio element.
            </audio>
          </div>
        )}
      </div>
    </div>
  );
}
