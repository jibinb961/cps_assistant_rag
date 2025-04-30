import { NextRequest, NextResponse } from 'next/server';
import { sessionDb } from '@/lib/supabase/database';

// POST /api/sessions/[sessionId]/context - Set session context
export async function POST(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  try {
    // Explicitly await params before accessing its properties
    const paramsData = await Promise.resolve(params);
    const sessionId = paramsData.sessionId;
    
    const body = await request.json();
    
    await sessionDb.setSessionContext({
      sessionId,
      programName: body.programName,
      summary: body.summary
    });
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error(`Error setting context:`, error);
    return NextResponse.json(
      { error: 'Failed to set session context' },
      { status: 500 }
    );
  }
} 