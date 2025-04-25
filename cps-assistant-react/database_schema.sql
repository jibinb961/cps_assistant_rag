-- chat_sessions table
CREATE TABLE public.chat_sessions (
  id uuid PRIMARY KEY,
  session_title text NOT NULL DEFAULT 'New Conversation',
  program_context text,
  summary text,
  created_at timestamp with time zone DEFAULT timezone('utc', now()),
  last_active timestamp with time zone DEFAULT timezone('utc', now())
);

-- chat_messages table
CREATE TABLE public.chat_messages (
  id uuid PRIMARY KEY,
  session_id uuid NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
  sender text NOT NULL CHECK (sender IN ('user', 'assistant', 'system')),
  message text NOT NULL,
  summary text,
  metadata jsonb,
  created_at timestamp with time zone DEFAULT timezone('utc', now())
);

-- Create indexes for better performance
CREATE INDEX chat_messages_session_id_idx ON public.chat_messages(session_id);
CREATE INDEX chat_sessions_last_active_idx ON public.chat_sessions(last_active);

-- Create RLS policies for chat_sessions
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public access to chat_sessions"
  ON public.chat_sessions
  FOR ALL
  USING (true);

-- Create RLS policies for chat_messages
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public access to chat_messages"
  ON public.chat_messages
  FOR ALL
  USING (true); 