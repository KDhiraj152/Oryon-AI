import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import type { ProcessRequest } from '../types/api';

const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB chunks

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [gradeLevel, setGradeLevel] = useState(8);
  const [subject, setSubject] = useState('Mathematics');
  const [targetLanguages, setTargetLanguages] = useState<string[]>(['Hindi']);
  const [outputFormat, setOutputFormat] = useState<'text' | 'audio' | 'both'>('both');
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [filePath, setFilePath] = useState<string | null>(null);

  const languages = ['Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi', 'Gujarati', 'Kannada', 'Malayalam', 'Punjabi', 'Odia'];
  const subjects = ['Mathematics', 'Science', 'Social Studies', 'English', 'Hindi', 'Computer Science'];
  const grades = [5, 6, 7, 8, 9, 10, 11, 12];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.type !== 'application/pdf') {
        setError('Please select a PDF file');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setError(null);
      setFilePath(null);
    }
  };

  const uploadFileInChunks = async (file: File): Promise<string> => {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const uploadId = `${Date.now()}_${file.name}`;

    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
      const start = chunkIndex * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const response = await api.uploadChunk(
        chunk,
        file.name,
        uploadId,
        chunkIndex,
        totalChunks,
        undefined,
        (progress) => {
          const totalProgress = ((chunkIndex + progress / 100) / totalChunks) * 100;
          setUploadProgress(Math.round(totalProgress));
        }
      );

      if (response.status === 'complete') {
        return `data/uploads/${file.name}`;
      }
    }

    throw new Error('Upload incomplete');
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      let uploadedFilePath: string;

      if (file.size > 10 * 1024 * 1024) {
        // Use chunked upload for files > 10MB
        uploadedFilePath = await uploadFileInChunks(file);
      } else {
        // Regular upload for smaller files
        const response = await api.uploadFile(file, (progress) => {
          setUploadProgress(Math.round(progress));
        });
        uploadedFilePath = response.file_path;
      }

      setFilePath(uploadedFilePath);
      setUploadProgress(100);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleProcess = async () => {
    if (!filePath) {
      setError('Please upload a file first');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const processRequest: ProcessRequest = {
        file_path: filePath,
        grade_level: gradeLevel,
        subject,
        target_languages: targetLanguages,
        output_format: outputFormat,
        validation_threshold: 0.8
      };

      const response = await api.processContent(processRequest);
      
      // Navigate to task tracking page
      navigate(`/tasks/${response.task_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Processing failed');
    } finally {
      setIsProcessing(false);
    }
  };



  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
          Upload Content
        </h1>
        <p className="text-gray-600 mb-8">Upload educational content for AI-powered processing</p>

        <div className="glass-card p-8 mb-6">
          <h2 className="text-2xl font-semibold mb-6">Upload PDF File</h2>
          
          <div className="space-y-6">
            <div>
              <label htmlFor="file-upload" className="block text-sm font-medium text-gray-700 mb-2">
                Select PDF File
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
              />
              {file && (
                <p className="mt-2 text-sm text-gray-600">
                  Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>

            {file && !filePath && (
              <button
                onClick={handleUpload}
                disabled={isUploading}
                className="btn-primary disabled:opacity-50"
              >
                {isUploading ? `Uploading... ${uploadProgress}%` : 'Upload File'}
              </button>
            )}

            {isUploading && (
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-primary-600 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            )}

            {filePath && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg text-green-700">
                ✓ File uploaded successfully
              </div>
            )}
          </div>
        </div>

        {filePath && (
          <div className="glass-card p-8 mb-6">
            <h2 className="text-2xl font-semibold mb-6">Processing Configuration</h2>
            
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="grade" className="block text-sm font-medium text-gray-700 mb-2">
                    Grade Level
                  </label>
                  <select
                    id="grade"
                    value={gradeLevel}
                    onChange={(e) => setGradeLevel(Number(e.target.value))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  >
                    {grades.map(grade => (
                      <option key={grade} value={grade}>Grade {grade}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-2">
                    Subject
                  </label>
                  <select
                    id="subject"
                    value={subject}
                    onChange={(e) => setSubject(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  >
                    {subjects.map(subj => (
                      <option key={subj} value={subj}>{subj}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label htmlFor="target-language" className="block text-sm font-medium text-gray-700 mb-2">
                  Target Language
                </label>
                <select
                  id="target-language"
                  value={targetLanguages[0]}
                  onChange={(e) => setTargetLanguages([e.target.value])}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                >
                  {languages.map(lang => (
                    <option key={lang} value={lang}>{lang}</option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="output-format" className="block text-sm font-medium text-gray-700 mb-2">
                  Output Format
                </label>
                <select
                  id="output-format"
                  value={outputFormat}
                  onChange={(e) => setOutputFormat(e.target.value as 'text' | 'audio' | 'both')}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                >
                  <option value="text">Text Only</option>
                  <option value="audio">Audio Only</option>
                  <option value="both">Text + Audio</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 mb-6 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {filePath && (
          <button
            onClick={handleProcess}
            disabled={isProcessing || targetLanguages.length === 0}
            className="w-full btn-primary py-3 text-lg font-semibold disabled:opacity-50"
          >
            {isProcessing ? 'Starting Processing...' : 'Process Content'}
          </button>
        )}
      </div>
    </div>
  );
}
