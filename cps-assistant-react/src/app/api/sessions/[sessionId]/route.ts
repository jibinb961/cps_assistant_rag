import { NextRequest, NextResponse } from 'next/server';
import { sessionDb } from '@/lib/supabase/database';

// GET /api/sessions/[sessionId] - Get a specific session
export async function GET(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  try {
    // Explicitly await params before accessing its properties
    const paramsData = await Promise.resolve(params);
    const sessionId = paramsData.sessionId;
    
    const session = await sessionDb.getSession(sessionId);
    
    return NextResponse.json(session);
  } catch (error) {
    console.error(`Error fetching session:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch session' },
      { status: 500 }
    );
  }
}

// PATCH /api/sessions/[sessionId] - Update a session
export async function PATCH(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  try {
    // Explicitly await params before accessing its properties
    const paramsData = await Promise.resolve(params);
    const sessionId = paramsData.sessionId;
    
    const body = await request.json();
    
    const updatedSession = await sessionDb.updateSession(sessionId, body);
    
    return NextResponse.json(updatedSession);
  } catch (error) {
    console.error(`Error updating session:`, error);
    return NextResponse.json(
      { error: 'Failed to update session' },
      { status: 500 }
    );
  }
} 