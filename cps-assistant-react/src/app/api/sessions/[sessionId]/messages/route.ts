import { NextRequest, NextResponse } from 'next/server';
import { messageDb, searchDb } from '@/lib/supabase/database';
import { generateChatResponse } from '@/lib/llm/groq';
import { Message } from '@/types';

// GET /api/sessions/[sessionId]/messages - Get messages for a session
export async function GET(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  try {
    // Explicitly await params before accessing its properties
    const paramsData = await Promise.resolve(params);
    const sessionId = paramsData.sessionId;
    
    const messages = await messageDb.getMessages(sessionId);
    
    return NextResponse.json(messages);
  } catch (error) {
    console.error(`Error fetching messages:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch messages' },
      { status: 500 }
    );
  }
}

// POST /api/sessions/[sessionId]/messages - Send a new message
export async function POST(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  try {
    // Explicitly await params before accessing its properties
    const paramsData = await Promise.resolve(params);
    const sessionId = paramsData.sessionId;
    
    const body = await request.json();
    const userMessage = body.content;
    
    if (!userMessage) {
      return NextResponse.json(
        { error: 'Message content is required' },
        { status: 400 }
      );
    }
    
    // Store user message
    await messageDb.createMessage(sessionId, 'user', userMessage);
    
    // Get relevant context through search
    const searchResults = await searchDb.searchPrograms({
      query: userMessage,
      // If there's program context, we could include it here
    });
    
    // Get conversation history (for context)
    const messageHistory = await messageDb.getMessages(sessionId);
    const chatHistory = messageHistory.map((msg: Message) => ({
      role: msg.role,
      content: msg.content
    }));
    
    // Generate AI response using Groq
    const aiResponse = await generateChatResponse(
      userMessage,
      searchResults,
      chatHistory.slice(-6) // Use last 6 messages for context
    );
    
    // Store AI response
    const savedResponse = await messageDb.createMessage(sessionId, 'assistant', aiResponse);
    
    return NextResponse.json(savedResponse);
  } catch (error) {
    console.error(`Error processing message:`, error);
    return NextResponse.json(
      { error: 'Failed to process message' },
      { status: 500 }
    );
  }
} 