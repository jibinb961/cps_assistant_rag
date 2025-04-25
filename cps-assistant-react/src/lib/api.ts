import { Message, ChatSession, SearchRequest, MessageRequest, SessionContext } from '../types';

// Base API endpoint
const API_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

// API error handling
class ApiError extends Error {
  status: number;
  
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

// Helper function for API requests
async function fetchAPI<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_URL}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(
      error.message || 'An error occurred while fetching data',
      response.status
    );
  }

  return response.json();
}

// Chat API services
export const chatApi = {
  // Sessions
  getSessions: async (): Promise<ChatSession[]> => {
    return fetchAPI<ChatSession[]>('/sessions');
  },
  
  createSession: async (title?: string): Promise<ChatSession> => {
    return fetchAPI<ChatSession>('/sessions', {
      method: 'POST',
      body: JSON.stringify({ title }),
    });
  },
  
  updateSession: async (sessionId: string, updates: Partial<ChatSession>): Promise<ChatSession> => {
    return fetchAPI<ChatSession>(`/sessions/${sessionId}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    });
  },
  
  // Messages
  getMessages: async (sessionId: string): Promise<Message[]> => {
    return fetchAPI<Message[]>(`/sessions/${sessionId}/messages`);
  },
  
  sendMessage: async (request: MessageRequest): Promise<Message> => {
    return fetchAPI<Message>(`/sessions/${request.sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content: request.message }),
    });
  },
  
  // Search
  search: async (request: SearchRequest): Promise<string[]> => {
    return fetchAPI<string[]>('/search', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },
  
  // Context
  setSessionContext: async (context: SessionContext): Promise<void> => {
    return fetchAPI<void>(`/sessions/${context.sessionId}/context`, {
      method: 'POST',
      body: JSON.stringify({ 
        programName: context.programName,
        summary: context.summary
      }),
    });
  },
  
  getAvailablePrograms: async (): Promise<string[]> => {
    return fetchAPI<string[]>('/programs');
  },
}; 