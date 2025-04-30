# CPS AI Assistant - Detailed Technical Explanation

This document provides a comprehensive explanation of the CPS AI Assistant React project, breaking down each component, the underlying technologies, and the workflow of the application.

## Table of Contents
1. [React and Next.js Fundamentals](#react-and-nextjs-fundamentals)
2. [Project Overview](#project-overview)
3. [Project Architecture](#project-architecture)
4. [File Structure Explained](#file-structure-explained)
5. [Key Components Explained](#key-components-explained)
6. [State Management](#state-management)
7. [API Routes and Backend](#api-routes-and-backend)
8. [Database Integration](#database-integration)
9. [LLM Integration](#llm-integration)
10. [Full Application Workflow](#full-application-workflow)

## React and Next.js Fundamentals

### What is React?
React is a JavaScript library for building user interfaces, primarily for single-page applications. It allows developers to create reusable UI components and efficiently update the DOM when data changes.

Key React concepts:
- **Components**: Self-contained, reusable pieces of UI
- **Props**: Data passed down from parent to child components
- **State**: Data that changes over time within a component
- **JSX**: A syntax extension that looks like HTML but works within JavaScript

### What is Next.js?
Next.js is a React framework that provides additional structure, features, and optimizations:

- **Server-side rendering**: Pages can be pre-rendered on the server
- **API Routes**: Backend API functionality built into the same project
- **File-based routing**: Pages are created based on file structure
- **App Directory**: The modern Next.js project structure with layouts and server components

In our project, we use the Next.js App Router pattern, where each route is defined in the `app` directory.

## Project Overview

The CPS AI Assistant is a chat interface that allows users to ask questions about Northeastern University's College of Professional Studies programs. It features:

1. A modern chat UI with message history
2. Session management to save conversations
3. Program-specific context for more accurate responses
4. Integration with Supabase for storage and vector search
5. LLM-powered responses using Groq API

## Project Architecture

The application follows a layered architecture:

1. **UI Layer**: React components for the chat interface
2. **State Management Layer**: Zustand store for global state
3. **API Layer**: Next.js API routes to handle requests
4. **Database Layer**: Supabase for storage and vector search
5. **LLM Layer**: Groq API for AI responses

Data flows through these layers as follows:
- User interacts with UI → State updated → API calls made → Database operations → LLM processing → Response displayed

## File Structure Explained

```
cps-assistant-react/
├── app/                   # Next.js app directory
│   ├── api/               # Backend API routes
│   │   ├── programs/      # Endpoints for available programs
│   │   ├── search/        # Endpoints for search functionality
│   │   └── sessions/      # Endpoints for managing sessions and messages
│   └── page.tsx           # Main application page
├── components/            # React UI components
│   ├── chat/              # Chat-related components
│   ├── layout/            # Layout components
│   ├── sidebar/           # Sidebar components
│   └── ui/                # Shared UI components (shadcn)
├── lib/                   # Utility libraries
│   ├── llm/               # LLM integration
│   ├── store/             # State management
│   └── supabase/          # Database integration
└── types/                 # TypeScript type definitions
```

## Key Components Explained

### Frontend Components

#### `app/page.tsx`
This is the entry point of the application. In Next.js, the file at `app/page.tsx` becomes the root route ('/') of your application. This file imports the `ChatLayout` component and renders it as the main UI.

```typescript
'use client';  // This directive enables client-side features

import { ChatLayout } from '../components/layout/chat-layout';

export default function Home() {
  return (
    <main className="h-screen">
      <ChatLayout />
    </main>
  );
}
```

- `'use client'`: A directive that tells Next.js this is a client component (runs in the browser)
- `ChatLayout`: The main layout component that contains the entire chat interface

#### `components/layout/chat-layout.tsx`
This component is the main layout container for the chat interface. It:
1. Manages the sidebar and main chat area
2. Loads sessions and messages from the API
3. Handles session selection and creation

```typescript
export const ChatLayout: React.FC = () => {
  // State management through Zustand hooks
  const sessions = useChatStore((state) => state.sessions);
  const activeSessionId = useChatStore((state) => state.activeSessionId);
  
  // Effect hooks to load data when the component mounts
  useEffect(() => {
    // Fetch sessions on initial load
  }, []);
  
  // Event handlers for session management
  const handleNewSession = async () => {/* ... */};
  const handleSelectSession = (sessionId: string) => {/* ... */};
  
  // Render the UI with sidebar and main content area
  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="w-64 h-full border-r bg-muted/40 dark:bg-muted/20">
        <SessionList /* ... */ />
      </div>
      
      {/* Main content */}
      <div className="flex-1 h-full">
        {activeSessionId ? (
          <ChatContainer /* ... */ />
        ) : (
          <div>Welcome message...</div>
        )}
      </div>
    </div>
  );
};
```

Key concepts:
- `useState` and `useEffect` are React hooks for managing state and side effects
- The component fetches data when it mounts and when dependencies change
- It conditionally renders different UI based on the state

#### `components/chat/chat-container.tsx`
This component manages the chat messages, input, and disambiguation modal. It:
1. Displays the chat message history
2. Handles sending new messages
3. Shows typing indicators during loading
4. Manages program disambiguation when needed

```typescript
export const ChatContainer: React.FC<ChatContainerProps> = ({
  sessionId,
  messages,
  isLoading,
  programContext
}) => {
  // State management
  const showDisambiguation = useChatStore((state) => state.showDisambiguation);
  
  // Handle sending messages
  const handleSendMessage = async (content: string) => {
    // Add user message to UI
    // Send message to API
    // Handle response
  };
  
  // Handle program selection
  const handleProgramSelection = async (program: string) => {
    // Set program context
    // Send pending message if exists
  };
  
  return (
    <div className="flex flex-col h-full">
      {/* Program context indicator */}
      {/* Chat messages area */}
      {/* Chat input */}
      {/* Disambiguation modal */}
    </div>
  );
};
```

#### `components/chat/message.tsx`
This component renders individual chat messages with proper styling, avatars, and markdown rendering:

```typescript
export const Message: React.FC<MessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      {/* Avatar */}
      {/* Message content with markdown */}
      {/* Timestamp */}
    </div>
  );
};
```

#### `components/chat/chat-input.tsx`
Handles user input for sending messages:

```typescript
export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled = false,
  placeholder = 'Ask about CPS programs...'
}) => {
  const [message, setMessage] = useState('');

  // Handle sending messages
  const handleSendMessage = () => {/* ... */};
  
  // Handle keyboard input (Enter to send)
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {/* ... */};
  
  return (
    <div className="flex items-end gap-2 border-t bg-background p-4">
      {/* Text input area */}
      {/* Send button */}
    </div>
  );
};
```

#### `components/chat/disambiguation-modal.tsx`
This modal appears when a user's query matches multiple programs, allowing them to select a specific program:

```typescript
export const DisambiguationModal: React.FC<DisambiguationModalProps> = ({
  open,
  onOpenChange,
  programs,
  onSelectProgram,
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Choose a Program</DialogTitle>
          <DialogDescription>
            Your question matches multiple programs. Please select...
          </DialogDescription>
        </DialogHeader>
        
        <ScrollArea>
          {/* Program selection buttons */}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};
```

#### `components/sidebar/session-list.tsx`
This component renders the list of chat sessions in the sidebar:

```typescript
export const SessionList: React.FC<SessionListProps> = ({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession
}) => {
  return (
    <div className="h-full flex flex-col">
      {/* New chat button */}
      <Separator />
      
      <ScrollArea>
        {/* List of chat sessions */}
      </ScrollArea>
    </div>
  );
};
```

### State Management

#### `lib/store/chat-store.ts`
This file uses Zustand to create a global state store for the application. Zustand is a lightweight state management library for React that's simpler than Redux but still powerful:

```typescript
export const useChatStore = create<ChatState>((set) => ({
  // Active session
  activeSessionId: null,
  setActiveSessionId: (sessionId) => set({ activeSessionId: sessionId }),
  
  // Sessions
  sessions: [],
  setSessions: (sessions) => set({ sessions }),
  addSession: (session) => set((state) => ({/* ... */})),
  
  // Messages
  messages: {},
  setMessages: (sessionId, messages) => set((state) => ({/* ... */})),
  addMessage: (sessionId, message) => set((state) => ({/* ... */})),
  
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
```

The store contains:
- Session management state (active session, list of sessions)
- Message storage for each session
- UI state (loading indicators)
- Program context for relevant searches
- Disambiguation state for handling ambiguous queries

### API Interaction

#### `lib/api.ts`
This file provides a clean interface for interacting with the backend API:

```typescript
export const chatApi = {
  // Sessions
  getSessions: async (): Promise<ChatSession[]> => {/* ... */},
  createSession: async (title?: string): Promise<ChatSession> => {/* ... */},
  updateSession: async (sessionId: string, updates: Partial<ChatSession>): Promise<ChatSession> => {/* ... */},
  
  // Messages
  getMessages: async (sessionId: string): Promise<Message[]> => {/* ... */},
  sendMessage: async (request: MessageRequest): Promise<Message> => {/* ... */},
  
  // Search
  search: async (request: SearchRequest): Promise<string[]> => {/* ... */},
  
  // Context
  setSessionContext: async (context: SessionContext): Promise<void> => {/* ... */},
  getAvailablePrograms: async (): Promise<string[]> => {/* ... */},
};
```

This provides a clean, typed interface for all API operations, handling the underlying fetch requests.

## API Routes and Backend

### `app/api/sessions/route.ts`
This file defines the API endpoints for managing sessions:

```typescript
// GET /api/sessions - Get all sessions
export async function GET() {
  try {
    const sessions = await sessionDb.getSessions();
    return NextResponse.json(sessions);
  } catch (error) {
    // Error handling
  }
}

// POST /api/sessions - Create a new session
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const title = body.title || 'New Conversation';
    
    const session = await sessionDb.createSession(title);
    return NextResponse.json(session);
  } catch (error) {
    // Error handling
  }
}
```

In Next.js, API routes are defined by files in the `app/api` directory. Each file exports HTTP method functions (GET, POST, etc.) that handle requests.

### `app/api/sessions/[sessionId]/messages/route.ts`
This file handles retrieving and sending messages for a specific session:

```typescript
// GET /api/sessions/[sessionId]/messages - Get messages for a session
export async function GET(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {/* ... */}

// POST /api/sessions/[sessionId]/messages - Send a new message
export async function POST(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  try {
    // Extract data from request
    const sessionId = params.sessionId;
    const body = await request.json();
    const userMessage = body.content;
    
    // Store user message
    await messageDb.createMessage(sessionId, 'user', userMessage);
    
    // Get relevant context through search
    const searchResults = await searchDb.searchPrograms({/* ... */});
    
    // Get conversation history
    const messageHistory = await messageDb.getMessages(sessionId);
    
    // Generate AI response
    const aiResponse = await generateChatResponse(/* ... */);
    
    // Store AI response
    const savedResponse = await messageDb.createMessage(/* ... */);
    
    return NextResponse.json(savedResponse);
  } catch (error) {
    // Error handling
  }
}
```

The dynamic route segment `[sessionId]` allows this API to handle any session ID.

### `app/api/search/route.ts`
This endpoint handles searching for program information:

```typescript
// POST /api/search - Search for programs/content
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query, programName, top_k } = body;
    
    // Validation
    if (!query) {
      return NextResponse.json(
        { error: 'Search query is required' },
        { status: 400 }
      );
    }
    
    // Perform search
    const searchResults = await searchDb.searchPrograms({
      query,
      programName,
      top_k
    });
    
    return NextResponse.json(searchResults);
  } catch (error) {
    // Error handling
  }
}
```

### `app/api/programs/route.ts`
This endpoint retrieves the list of available programs:

```typescript
// GET /api/programs - Get available programs
export async function GET() {
  try {
    const programs = await searchDb.getAvailablePrograms();
    return NextResponse.json(programs);
  } catch (error) {
    // Error handling
  }
}
```

## Database Integration

### `lib/supabase/client.ts`
This file initializes the Supabase client:

```typescript
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseKey);
```

### `lib/supabase/database.ts`
This file contains all the database operations, organized into modules:

```typescript
// Session-related database operations
export const sessionDb = {
  getSessions: async (): Promise<ChatSession[]> => {/* ... */},
  getSession: async (sessionId: string): Promise<ChatSession> => {/* ... */},
  createSession: async (title: string = 'New Conversation'): Promise<ChatSession> => {/* ... */},
  updateSession: async (sessionId: string, updates: Partial<ChatSession>): Promise<ChatSession> => {/* ... */},
  setSessionContext: async (context: SessionContext): Promise<void> => {/* ... */},
};

// Message-related database operations
export const messageDb = {
  getMessages: async (sessionId: string): Promise<Message[]> => {/* ... */},
  createMessage: async (sessionId: string, role: 'user' | 'assistant' | 'system', content: string): Promise<Message> => {/* ... */}
};

// Search-related database operations
export const searchDb = {
  searchPrograms: async (request: SearchRequest): Promise<SearchResult[]> => {/* ... */},
  getAvailablePrograms: async (): Promise<string[]> => {/* ... */}
};
```

The database schema (`database_schema.sql`) defines two main tables:
1. `chat_sessions` - Stores information about chat sessions
2. `chat_messages` - Stores all messages within sessions

## LLM Integration

### `lib/llm/groq.ts`
This file handles interaction with the Groq API for generating AI responses:

```typescript
// Function to generate content using Groq API
export async function generateChatResponse(
  query: string,
  context: SearchResult[],
  chat_history: GroqMessage[] = []
): Promise<string> {
  // Prepare context from search results
  const contextText = context.map(/* ... */).join('\n\n');

  // Create system message with instructions and context
  const systemMessage = {/* ... */};

  // Combine system message, chat history, and query
  const messages = [systemMessage, ...chat_history, { role: 'user', content: query }];

  // Make API request to Groq
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {/* ... */});
  
  // Return the generated response
  return data.choices[0].message.content;
}
```

This function:
1. Takes a user query, search context, and chat history
2. Formats them into a prompt for the LLM
3. Sends a request to the Groq API
4. Returns the generated response

## Full Application Workflow

Here's how all these pieces work together:

1. **Initial Load**:
   - User visits the application
   - Next.js renders the page
   - `ChatLayout` component loads and fetches existing sessions
   - If sessions exist, the first one is selected and its messages are loaded

2. **Starting a New Chat**:
   - User clicks "New Chat" in the sidebar
   - `handleNewSession` in `ChatLayout` creates a new session in the database
   - The new session becomes active, and the chat interface is displayed

3. **Sending a Message**:
   - User types a message and presses Enter
   - `ChatInput` component calls `handleSendMessage` in `ChatContainer`
   - The message is added to the UI immediately (optimistic update)
   - An API request is sent to `/api/sessions/[sessionId]/messages`
   - The API route stores the user message in the database
   - The API route performs a vector search for relevant content
   - The API route calls the Groq API with the user's message and context
   - The API route stores the AI response in the database
   - The response is returned to the frontend and displayed in the chat

4. **Program Disambiguation**:
   - If the search results indicate multiple possible programs
   - The disambiguation modal is displayed
   - User selects a specific program
   - The program context is stored for the session
   - The original message is processed with this context

5. **Switching Sessions**:
   - User clicks on a different session in the sidebar
   - `handleSelectSession` in `ChatLayout` changes the active session
   - Messages for the new session are loaded
   - The chat interface updates to show the selected conversation

## Summary

The CPS AI Assistant is a modern React application built with Next.js that provides a chat interface for querying information about Northeastern University's CPS programs. It uses:

- React components for the UI
- Zustand for state management
- Next.js API routes for backend functionality
- Supabase for database storage and vector search
- Groq API for LLM-powered responses

The application is designed to be modular, with clear separation of concerns between UI components, state management, API interactions, and database operations. 