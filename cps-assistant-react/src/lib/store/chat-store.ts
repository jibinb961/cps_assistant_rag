import { create } from 'zustand';
import { ChatSession, Message } from '../../types';

interface ChatState {
  // Active session
  activeSessionId: string | null;
  setActiveSessionId: (sessionId: string | null) => void;
  
  // Sessions
  sessions: ChatSession[];
  setSessions: (sessions: ChatSession[]) => void;
  addSession: (session: ChatSession) => void;
  updateSession: (sessionId: string, updates: Partial<ChatSession>) => void;
  
  // Messages
  messages: Record<string, Message[]>;
  setMessages: (sessionId: string, messages: Message[]) => void;
  addMessage: (sessionId: string, message: Message) => void;
  
  // UI state
  isLoading: boolean;
  setLoading: (loading: boolean) => void;
  
  // Program context
  programContext: string | null;
  setProgramContext: (program: string | null) => void;
  
  // Disambiguation
  programOptions: string[];
  setProgramOptions: (options: string[]) => void;
  showDisambiguation: boolean;
  setShowDisambiguation: (show: boolean) => void;
  pendingMessage: string | null;
  setPendingMessage: (message: string | null) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  // Active session
  activeSessionId: null,
  setActiveSessionId: (sessionId) => set({ activeSessionId: sessionId }),
  
  // Sessions
  sessions: [],
  setSessions: (sessions) => set({ sessions }),
  addSession: (session) => set((state) => ({ 
    sessions: [...state.sessions, session] 
  })),
  updateSession: (sessionId, updates) => set((state) => ({
    sessions: state.sessions.map(session => 
      session.id === sessionId ? { ...session, ...updates } : session
    )
  })),
  
  // Messages
  messages: {},
  setMessages: (sessionId, messages) => set((state) => ({
    messages: { ...state.messages, [sessionId]: messages }
  })),
  addMessage: (sessionId, message) => set((state) => ({
    messages: { 
      ...state.messages, 
      [sessionId]: [...(state.messages[sessionId] || []), message] 
    }
  })),
  
  // UI state
  isLoading: false,
  setLoading: (loading) => set({ isLoading: loading }),
  
  // Program context
  programContext: null,
  setProgramContext: (program) => set({ programContext: program }),
  
  // Disambiguation
  programOptions: [],
  setProgramOptions: (options) => set({ programOptions: options }),
  showDisambiguation: false,
  setShowDisambiguation: (show) => set({ showDisambiguation: show }),
  pendingMessage: null,
  setPendingMessage: (message) => set({ pendingMessage: message }),
})); 