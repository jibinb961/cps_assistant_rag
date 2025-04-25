# CPS AI Assistant

A modern React application that serves as an AI assistant for Northeastern University's College of Professional Studies. The application allows students to chat with an AI agent to retrieve program information, ask questions, and receive context-aware, relevant answers.

## Features

- Modern chat interface with markdown rendering and timestamp display
- Session management for preserving conversations
- Program disambiguation for handling queries related to multiple programs
- Context-aware responses using Retrieval Augmented Generation (RAG)
- Integration with Supabase for storage and vector search
- LLM-powered responses via Groq API

## Technologies Used

- Frontend: React, TypeScript, Next.js
- UI Framework: shadcn/ui
- State Management: Zustand
- Backend: Next.js API Routes
- Database: Supabase (PostgreSQL)
- AI: Groq LLM API

## Prerequisites

- Node.js 18+ and npm
- Supabase account and project
- Groq API key

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env.local` file based on the `env.example` template:
   ```
   # Supabase credentials
   NEXT_PUBLIC_SUPABASE_URL=your_supabase_url_here
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key_here
   
   # Groq API key for LLM interactions
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Set up your Supabase database using the SQL in `database_schema.sql`

5. Run the development server:
   ```bash
   npm run dev
   ```

6. Open [http://localhost:3000](http://localhost:3000) in your browser

## Database Setup

1. Create a new Supabase project
2. Run the SQL queries from the `database_schema.sql` file in the SQL editor
3. Ensure your site_pages table already exists and has program_name fields (or modify the code accordingly)
4. Set up vector embeddings for your content if needed

## Usage

### Starting a New Chat

1. Click the "New Chat" button in the sidebar
2. Type your question in the input box at the bottom of the screen
3. If your question relates to multiple programs, you'll be prompted to select a specific program

### Managing Sessions

- All chats are preserved and can be accessed from the sidebar
- Click on any previous chat session to continue the conversation
- The program context is preserved between sessions

### Example Queries

- "What are the core courses for the Analytics program?"
- "Tell me about admission requirements for Computer Science"
- "How do co-op opportunities work?"
- "Compare the Analytics and Data Science programs"

## Development

### Project Structure

- `/app` - Next.js app directory including pages and API routes
- `/components` - React components 
- `/lib` - Utility functions, Supabase client, and Zustand store
- `/types` - TypeScript type definitions

### API Routes

- GET/POST `/api/sessions` - List or create sessions
- GET/PATCH `/api/sessions/[sessionId]` - Get or update a specific session
- GET/POST `/api/sessions/[sessionId]/messages` - Get messages or send a new message
- POST `/api/sessions/[sessionId]/context` - Set session context
- POST `/api/search` - Search for program information
- GET `/api/programs` - Get available programs

## License

MIT
