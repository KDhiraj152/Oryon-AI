import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import type { ProcessedContent } from '../types/api';

export default function LibraryPage() {
  const [content, setContent] = useState<ProcessedContent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filters, setFilters] = useState({
    grade: undefined as number | undefined,
    subject: '',
    language: ''
  });
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  const grades = [5, 6, 7, 8, 9, 10, 11, 12];
  const subjects = ['Mathematics', 'Science', 'Social Studies', 'English', 'Hindi', 'Computer Science'];
  const languages = ['Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi', 'Gujarati', 'Kannada', 'Malayalam', 'Punjabi', 'Odia'];

  useEffect(() => {
    fetchContent();
  }, [filters, offset]);

  const fetchContent = async () => {
    try {
      const response = await api.getLibrary({
        ...filters,
        limit: 20,
        offset
      });
      setContent(response.items);
      setHasMore(response.has_more);
    } catch (err) {
      console.error('Failed to load library:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
          Content Library
        </h1>

        <div className="glass-card p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Filters</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label htmlFor="filter-grade" className="block text-sm font-medium text-gray-700 mb-2">
                Grade Level
              </label>
              <select
                id="filter-grade"
                value={filters.grade || ''}
                onChange={(e) => setFilters({ ...filters, grade: e.target.value ? Number(e.target.value) : undefined })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                <option value="">All Grades</option>
                {grades.map(grade => (
                  <option key={grade} value={grade}>Grade {grade}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="filter-subject" className="block text-sm font-medium text-gray-700 mb-2">
                Subject
              </label>
              <select
                id="filter-subject"
                value={filters.subject}
                onChange={(e) => setFilters({ ...filters, subject: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                <option value="">All Subjects</option>
                {subjects.map(subject => (
                  <option key={subject} value={subject}>{subject}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="filter-language" className="block text-sm font-medium text-gray-700 mb-2">
                Language
              </label>
              <select
                id="filter-language"
                value={filters.language}
                onChange={(e) => setFilters({ ...filters, language: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                <option value="">All Languages</option>
                {languages.map(lang => (
                  <option key={lang} value={lang}>{lang}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
          </div>
        ) : content.length === 0 ? (
          <div className="glass-card p-12 text-center">
            <p className="text-gray-600">No content found</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              {content.map((item) => (
                <Link
                  key={item.id}
                  to={`/content/${item.id}`}
                  className="glass-card p-6 hover:scale-105 transition-transform"
                >
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">{item.subject}</h3>
                  <p className="text-sm text-gray-600 line-clamp-3 mb-4">
                    {item.simplified_text.substring(0, 200)}...
                  </p>
                  <div className="flex justify-between items-center text-sm">
                    <div className="flex gap-3 text-gray-500">
                      <span>Grade {item.grade_level}</span>
                      <span>•</span>
                      <span>{item.language}</span>
                    </div>
                    {item.audio_available && (
                      <div className="text-green-600">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                        </svg>
                      </div>
                    )}
                  </div>
                </Link>
              ))}
            </div>

            {hasMore && (
              <div className="text-center">
                <button
                  onClick={() => setOffset(offset + 20)}
                  className="btn-primary"
                >
                  Load More
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
