import { supabase } from './client';
import { ChatSession, Message, SearchRequest, SearchResult, SessionContext } from '../../types';
import { v4 as uuidv4 } from 'uuid';

// Session-related database operations
export const sessionDb = {
  // Get all chat sessions
  getSessions: async (): Promise<ChatSession[]> => {
    const { data, error } = await supabase
      .from('chat_sessions')
      .select('*')
      .order('last_active', { ascending: false });

    if (error) throw error;

    return data.map(session => ({
      id: session.id,
      title: session.session_title,
      createdAt: session.created_at,
      lastActive: session.last_active,
      programContext: session.program_context
    }));
  },

  // Get a specific session by ID
  getSession: async (sessionId: string): Promise<ChatSession> => {
    const { data, error } = await supabase
      .from('chat_sessions')
      .select('*')
      .eq('id', sessionId)
      .single();

    if (error) throw error;

    return {
      id: data.id,
      title: data.session_title,
      createdAt: data.created_at,
      lastActive: data.last_active,
      programContext: data.program_context
    };
  },

  // Create a new session
  createSession: async (title: string = 'New Conversation'): Promise<ChatSession> => {
    const id = uuidv4();
    const now = new Date().toISOString();

    const { data, error } = await supabase
      .from('chat_sessions')
      .insert({
        id,
        session_title: title,
        created_at: now,
        last_active: now
      })
      .select()
      .single();

    if (error) throw error;

    return {
      id: data.id,
      title: data.session_title,
      createdAt: data.created_at,
      lastActive: data.last_active,
      programContext: data.program_context
    };
  },

  // Update a session
  updateSession: async (sessionId: string, updates: Partial<ChatSession>): Promise<ChatSession> => {
    const updateData: any = {};
    
    if (updates.title) updateData.session_title = updates.title;
    if (updates.programContext) updateData.program_context = updates.programContext;
    
    // Always update last_active
    updateData.last_active = new Date().toISOString();

    const { data, error } = await supabase
      .from('chat_sessions')
      .update(updateData)
      .eq('id', sessionId)
      .select()
      .single();

    if (error) throw error;

    return {
      id: data.id,
      title: data.session_title,
      createdAt: data.created_at,
      lastActive: data.last_active,
      programContext: data.program_context
    };
  },

  // Set session context
  setSessionContext: async (context: SessionContext): Promise<void> => {
    const { error } = await supabase
      .from('chat_sessions')
      .update({
        program_context: context.programName,
        last_active: new Date().toISOString()
      })
      .eq('id', context.sessionId);

    if (error) throw error;
  },
};

// Message-related database operations
export const messageDb = {
  // Get messages for a session
  getMessages: async (sessionId: string): Promise<Message[]> => {
    const { data, error } = await supabase
      .from('chat_messages')
      .select('*')
      .eq('session_id', sessionId)
      .order('created_at', { ascending: true });

    if (error) throw error;

    return data.map(message => ({
      id: message.id,
      role: message.sender,
      content: message.message,
      createdAt: message.created_at
    }));
  },

  // Create a new message
  createMessage: async (
    sessionId: string,
    role: 'user' | 'assistant' | 'system',
    content: string
  ): Promise<Message> => {
    const id = uuidv4();
    const now = new Date().toISOString();

    const { data, error } = await supabase
      .from('chat_messages')
      .insert({
        id,
        session_id: sessionId,
        sender: role,
        message: content,
        created_at: now
      })
      .select()
      .single();

    if (error) throw error;

    // Update session last_active
    await supabase
      .from('chat_sessions')
      .update({ last_active: now })
      .eq('id', sessionId);

    return {
      id: data.id,
      role: data.sender,
      content: data.message,
      createdAt: data.created_at
    };
  }
};

// Search-related database operations
export const searchDb = {
  // Vector search for relevant content
  searchPrograms: async (request: SearchRequest): Promise<SearchResult[]> => {
    // Create embeddings for the query using local Ollama
    const embedding = await createEmbedding(request.query);
    
    // Determine the search mode based on program name
    let searchMode = 'general';
    if (request.programName === 'coop_information') {
      searchMode = 'coop';
    } else if (request.programName && request.programName !== '') {
      searchMode = 'specific';
    }
    
    // Default match parameters
    const params: any = {
      query_embedding: embedding,
      match_count: request.top_k || 5,
      search_mode: searchMode
    };
    
    // Add filter if program name is specified
    if (request.programName) {
      // Set the appropriate filter based on the search mode
      if (searchMode === 'coop') {
        params.filter = { 
          source: 'coop_information' 
        };
      } else if (searchMode === 'specific') {
        params.filter = {
          source: 'cps_program_docs',
          program_name: request.programName
        };
      } else {
        params.filter = { 
          source: 'cps_program_docs' 
        };
      }
    }
    
    try {
      console.log('🔍 Vector Search Query:', request.query);
      // Print params without the embedding vector to avoid cluttering the terminal
      const logParams = { ...params };
      delete logParams.query_embedding; // Remove embedding from log output
      console.log('🔧 Search Parameters:', JSON.stringify(logParams, null, 2));
      
      const { data, error } = await supabase.rpc(
        'match_site_pages',
        params
      );

      if (error) {
        console.error('❌ Error from supabase rpc call:', error);
        throw error;
      }

      console.log(`✅ Retrieved ${data.length} chunks from vector search`);
      
      // Log each chunk with relevant info but trim long content
      data.forEach((chunk: any, index: number) => {
        const trimmedContent = chunk.content ? 
          (chunk.content.length > 150 ? chunk.content.substring(0, 150) + '...' : chunk.content) : '';
        
        console.log(`
Chunk #${index + 1}:
Title: ${chunk.title || 'N/A'}
Similarity: ${(chunk.similarity * 100).toFixed(2)}%
Source: ${chunk.metadata?.source || 'N/A'}
Program: ${chunk.metadata?.program_name || 'N/A'}
Content Preview: ${trimmedContent}
-------------------`);
      });

      return data.map((item: any) => ({
        title: item.title || '',
        content: item.content || '',
        url: item.url || '',
        program_name: item.metadata?.program_name || '',
        similarity: item.similarity
      }));
    } catch (error) {
      console.error('❌ Error in searchPrograms:', error);
      throw error;
    }
  },

  // Get available programs
  getAvailablePrograms: async (): Promise<string[]> => {
    const { data, error } = await supabase
      .from('site_pages')
      .select('metadata')
      .not('metadata->program_name', 'is', null)
      .order('metadata->program_name');

    if (error) throw error;

    // Extract unique program names
    const programSet = new Set<string>();
    data.forEach(item => {
      const programName = item.metadata?.program_name;
      if (programName) {
        programSet.add(programName);
      }
    });

    return Array.from(programSet);
  }
};

// Helper function to create embeddings using local Ollama server
async function createEmbedding(text: string): Promise<number[]> {
  try {
    // Make a request to the local Ollama server
    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'nomic-embed-text:latest',
        prompt: text
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Embedding request failed with status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.embedding) {
      throw new Error('No embedding returned from Ollama');
    }
    
    console.log(`Generated embedding with ${data.embedding.length} dimensions`);
    
    return data.embedding;
  } catch (error) {
    console.error('Error generating embedding with Ollama:', error);
    // Fallback to random vector if Ollama server is not available
    console.warn('Falling back to random vector embedding');
    const dummyVector = Array(768).fill(0).map(() => Math.random());
    return dummyVector;
  }
} 