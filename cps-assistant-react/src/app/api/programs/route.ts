import { NextResponse } from 'next/server';
import { searchDb } from '@/lib/supabase/database';

// GET /api/programs - Get available programs
export async function GET() {
  try {
    const programs = await searchDb.getAvailablePrograms();
    return NextResponse.json(programs);
  } catch (error) {
    console.error('Error fetching programs:', error);
    return NextResponse.json(
      { error: 'Failed to fetch available programs' },
      { status: 500 }
    );
  }
} 