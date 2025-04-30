import { NextRequest, NextResponse } from 'next/server';
import { searchDb } from '@/lib/supabase/database';

// POST /api/search - Search for programs/content
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query, programName, top_k } = body;
    
    if (!query) {
      return NextResponse.json(
        { error: 'Search query is required' },
        { status: 400 }
      );
    }
    
    const searchResults = await searchDb.searchPrograms({
      query,
      programName,
      top_k
    });
    
    return NextResponse.json(searchResults);
  } catch (error) {
    console.error('Error searching programs:', error);
    return NextResponse.json(
      { error: 'Failed to search programs' },
      { status: 500 }
    );
  }
} 