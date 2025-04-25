// Chat Types
export type MessageRole = 'user' | 'assistant' | 'system';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  lastActive: string;
  programContext?: string;
}

// Database Types
export interface SitePage {
  id: number;
  title: string;
  url: string;
  content: string;
  program_name?: string;
  source: string;
  embedding?: number[];
}

// Search Types
export interface SearchResult {
  title: string;
  content: string;
  url?: string;
  program_name?: string;
  similarity?: number;
}

// API Types
export interface SearchRequest {
  query: string;
  programName?: string;
  top_k?: number;
}

export interface MessageRequest {
  sessionId: string;
  message: string;
}

export interface SessionContext {
  sessionId: string;
  programName?: string;
  summary?: string;
} 