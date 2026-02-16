/**
 * Chat utility functions extracted from Chat.tsx for modularity and reuse.
 */

import type { Message, Citation } from '../store';

// ─── File Upload Processing ─────────────────────────────────────────────────

interface ProcessedFiles {
  context: string;
  errors: string[];
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const ALLOWED_TEXT_TYPES = new Set([
  'text/plain', 'text/csv', 'text/markdown',
  'application/json', 'application/xml',
  'application/pdf',
]);

function isTextReadable(file: File): boolean {
  return (
    file.type.startsWith('text/') ||
    ALLOWED_TEXT_TYPES.has(file.type) ||
    file.name.endsWith('.md') ||
    file.name.endsWith('.txt') ||
    file.name.endsWith('.csv') ||
    file.name.endsWith('.json')
  );
}

async function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
    reader.readAsText(file);
  });
}

export async function processUploadedFiles(files: File[]): Promise<ProcessedFiles> {
  const errors: string[] = [];
  const contextParts: string[] = [];

  for (const file of files) {
    if (file.size > MAX_FILE_SIZE) {
      errors.push(`${file.name} exceeds 10 MB limit`);
      continue;
    }

    if (file.type.startsWith('image/')) {
      contextParts.push(`[Attached image: ${file.name}]`);
      continue;
    }

    if (file.type.startsWith('audio/') || file.type.startsWith('video/')) {
      contextParts.push(`[Attached ${file.type.split('/')[0]}: ${file.name}]`);
      continue;
    }

    if (isTextReadable(file)) {
      try {
        const text = await readFileAsText(file);
        const truncated = text.length > 8000 ? text.slice(0, 8000) + '\n...(truncated)' : text;
        contextParts.push(`--- ${file.name} ---\n${truncated}\n--- end ${file.name} ---`);
      } catch {
        errors.push(`Could not read ${file.name}`);
      }
    } else {
      contextParts.push(`[Attached file: ${file.name} (${file.type || 'unknown type'})]`);
    }
  }

  return { context: contextParts.join('\n\n'), errors };
}

// ─── Message Factories ──────────────────────────────────────────────────────

export function createUserMessage(
  conversationId: string,
  content: string,
  files?: File[],
): Message {
  return {
    id: Date.now().toString(),
    conversationId,
    role: 'user',
    content,
    timestamp: new Date().toISOString(),
    attachments: files?.map((f) => ({
      name: f.name,
      url: URL.createObjectURL(f),
      type: f.type,
      size: f.size,
    })),
  };
}

export function createAssistantMessage(
  conversationId: string,
  content: string,
  isError: boolean,
  meta?: {
    citations?: Citation[];
    modelUsed?: string;
    latencyMs?: number;
    tokenCount?: number;
  },
): Message {
  return {
    id: (Date.now() + 1).toString(),
    conversationId,
    role: 'assistant',
    content,
    timestamp: new Date().toISOString(),
    isError,
    citations: meta?.citations,
    modelUsed: meta?.modelUsed,
    latencyMs: meta?.latencyMs,
    tokenCount: meta?.tokenCount,
  };
}

export function createErrorMessage(conversationId: string): Message {
  return {
    id: (Date.now() + 1).toString(),
    conversationId,
    role: 'assistant',
    content: 'Sorry, something went wrong. Please try again.',
    timestamp: new Date().toISOString(),
    isError: true,
  };
}

// ─── Request Builders ───────────────────────────────────────────────────────

export function buildChatHeaders(accessToken?: string): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (accessToken) {
    headers['Authorization'] = `Bearer ${accessToken}`;
    headers['Accept'] = 'text/event-stream';
  }
  return headers;
}

export function buildChatRequestBody({
  message,
  conversationId,
  history,
  language,
  isAuthenticated,
}: {
  message: string;
  conversationId: string;
  history: { role: string; content: string }[];
  language?: string;
  isAuthenticated: boolean;
}): string {
  if (isAuthenticated) {
    return JSON.stringify({
      message,
      conversation_id: conversationId,
      history,
      language,
    });
  }
  return JSON.stringify({ message, language });
}

export function buildConversationHistory(
  messages: Message[],
): { role: string; content: string }[] {
  return messages.slice(-10).map((m) => ({
    role: m.role,
    content: m.content,
  }));
}

export function getChatEndpoint(isAuthenticated: boolean): string {
  return isAuthenticated ? '/api/v2/chat' : '/api/v2/chat/guest';
}

export function getResponseLanguage(
  language: string | undefined,
  selectedLanguage: string,
): string | undefined {
  if (language && language !== 'auto') return language;
  if (selectedLanguage !== 'auto') return selectedLanguage;
  return undefined;
}
