import { NextResponse } from 'next/server';
import { sessionDb } from '@/lib/supabase/database';

// GET /api/sessions - Get all sessions
export async function GET() {
  try {
    const sessions = await sessionDb.getSessions();
    return NextResponse.json(sessions);
  } catch (error) {
    console.error('Error fetching sessions:', error);
    return NextResponse.json(
      { error: 'Failed to fetch sessions' },
      { status: 500 }
    );
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
    console.error('Error creating session:', error);
    return NextResponse.json(
      { error: 'Failed to create session' },
      { status: 500 }
    );
  }
} 