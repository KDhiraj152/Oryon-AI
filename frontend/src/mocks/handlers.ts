import { http, HttpResponse } from 'msw';
import type { ProcessedContent, PaginatedResponse } from '../types/api';

const API_BASE_URL = 'http://localhost:8000';

const mockLibraryItems: ProcessedContent[] = [
  {
    id: '1',
    original_text: 'Photosynthesis is the process by which plants make food using sunlight, water, and carbon dioxide.',
    simplified_text: 'Plants make their own food using sunlight, water, and air.',
    translations: {
      Hindi: 'पौधे सूर्य के प्रकाश, पानी और हवा का उपयोग करके अपना भोजन बनाते हैं।',
      Tamil: 'தாவரங்கள் சூரிய ஒளி, நீர் மற்றும் காற்றைப் பயன்படுத்தி தங்கள் உணவை உருவாக்குகின்றன.'
    },
    validation_score: 0.95,
    audio_available: true,
    grade_level: 6,
    subject: 'Science',
    language: 'English',
    created_at: '2025-11-15T10:00:00Z',
    updated_at: '2025-11-15T10:05:00Z'
  },
  {
    id: '2',
    original_text: 'The Pythagorean theorem states that in a right triangle, a² + b² = c².',
    simplified_text: 'In a right triangle, the square of the longest side equals the sum of squares of the other two sides.',
    translations: {
      Hindi: 'समकोण त्रिभुज में, सबसे लंबी भुजा का वर्ग अन्य दो भुजाओं के वर्गों के योग के बराबर होता है।',
      Tamil: 'செங்கோண முக்கோணத்தில், நீண்ட பக்கத்தின் சதுரம் மற்ற இரு பக்கங்களின் சதுரங்களின் கூட்டுத்தொகைக்கு சமம்.'
    },
    validation_score: 0.98,
    audio_available: true,
    grade_level: 8,
    subject: 'Mathematics',
    language: 'English',
    created_at: '2025-11-14T15:30:00Z',
    updated_at: '2025-11-14T15:35:00Z'
  }
];

export const handlers = [
  http.get(`${API_BASE_URL}/api/v1/library`, ({ request }) => {
    const url = new URL(request.url);
    const limit = Number.parseInt(url.searchParams.get('limit') || '20');
    const offset = Number.parseInt(url.searchParams.get('offset') || '0');
    const language = url.searchParams.get('language');
    const grade = url.searchParams.get('grade');
    const subject = url.searchParams.get('subject');

    let filteredItems = [...mockLibraryItems];
    
    if (language) {
      filteredItems = filteredItems.filter(item => 
        item.translations?.[language] !== undefined
      );
    }
    
    if (grade) {
      filteredItems = filteredItems.filter(item => 
        item.grade_level === Number.parseInt(grade)
      );
    }
    
    if (subject) {
      filteredItems = filteredItems.filter(item => 
        item.subject === subject
      );
    }

    const paginatedItems = filteredItems.slice(offset, offset + limit);
    
    const response: PaginatedResponse<ProcessedContent> = {
      items: paginatedItems,
      total: filteredItems.length,
      limit,
      offset,
      has_more: offset + limit < filteredItems.length
    };

    return HttpResponse.json(response);
  }),

  http.get(`${API_BASE_URL}/api/v1/content/search`, ({ request }) => {
    const url = new URL(request.url);
    const q = url.searchParams.get('q') || '';
    const limit = Number.parseInt(url.searchParams.get('limit') || '20');

    const results = mockLibraryItems.filter(item => {
      const searchText = q.toLowerCase();
      return (
        item.original_text.toLowerCase().includes(searchText) ||
        item.simplified_text.toLowerCase().includes(searchText) ||
        item.subject.toLowerCase().includes(searchText)
      );
    }).slice(0, limit);

    return HttpResponse.json({ results });
  }),
];
