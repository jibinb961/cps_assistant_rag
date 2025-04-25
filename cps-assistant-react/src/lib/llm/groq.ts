import { SearchResult } from '../../types';

// API key should be loaded from environment variables
const GROQ_API_KEY = process.env.GROQ_API_KEY;

interface GroqMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface GroqRequest {
  messages: GroqMessage[];
  model: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stream?: boolean;
}

interface GroqResponse {
  choices: {
    message: {
      role: string;
      content: string;
    };
    index: number;
    finish_reason: string;
  }[];
  id: string;
  model: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// Function to generate content using Groq API
export async function generateChatResponse(
  query: string,
  context: SearchResult[],
  chat_history: GroqMessage[] = []
): Promise<string> {
  if (!GROQ_API_KEY) {
    throw new Error('GROQ_API_KEY is not set in environment variables');
  }

  // Prepare the context text from search results
  const contextText = context
    .map(
      (result) =>
        `Title: ${result.title}\nContent: ${result.content}\n${
          result.program_name ? `Program: ${result.program_name}` : ''
        }`
    )
    .join('\n\n');

  // Create the system message with instructions and context
  const systemMessage: GroqMessage = {
    role: 'system',
    content: `You are an AI assistant for the College of Professional Studies at Northeastern University.
    Your goal is to provide helpful, accurate information about various programs, courses, and requirements.
    
    Below is relevant information from the university's database to help answer the user's question:
    
    ${contextText}
    
    When responding:
    1. Be helpful, concise, and accurate
    2. Format your response using markdown for better readability
    3. If the information provided isn't sufficient to answer the question fully, acknowledge the limitations
    4. Do not make up information that is not provided in the context
    5. If you're uncertain about something, state that clearly rather than guessing`,
  };

  // Combine system message, chat history, and current query
  const messages: GroqMessage[] = [
    systemMessage, 
    ...chat_history, 
    { role: 'user' as const, content: query }
  ];

  // Prepare the request for Groq API
  const requestBody: GroqRequest = {
    messages,
    model: 'llama3-70b-8192', // Using Llama 3 70B model
    temperature: 0.7,
    max_tokens: 1024,
    top_p: 0.9,
  };

  try {
    // Call the Groq API
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${GROQ_API_KEY}`,
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Groq API error: ${JSON.stringify(errorData)}`);
    }

    // Parse the response
    const data: GroqResponse = await response.json();
    
    // Return the generated content
    return data.choices[0].message.content;
  } catch (error) {
    console.error('Error calling Groq API:', error);
    throw error;
  }
}

// Function to create a summary for conversation context
export async function generateConversationSummary(
  messages: GroqMessage[]
): Promise<string> {
  if (!GROQ_API_KEY) {
    throw new Error('GROQ_API_KEY is not set in environment variables');
  }

  // Create a system message for summarization
  const systemMessage: GroqMessage = {
    role: 'system',
    content: `Summarize the following conversation between a user and an assistant about Northeastern University's College of Professional Studies programs.
    Create a brief summary that captures the main topics, programs discussed, and key questions asked.
    The summary should be concise (1-3 sentences) and will be used as context for future messages.`,
  };

  // Add a user message with the conversation to summarize
  const userMessage: GroqMessage = {
    role: 'user',
    content: messages.map(m => `${m.role}: ${m.content}`).join('\n\n'),
  };

  // Prepare the request for Groq API
  const requestBody: GroqRequest = {
    messages: [systemMessage, userMessage],
    model: 'llama3-70b-8192',
    temperature: 0.3,
    max_tokens: 256,
    top_p: 0.9,
  };

  try {
    // Call the Groq API
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${GROQ_API_KEY}`,
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Groq API error: ${JSON.stringify(errorData)}`);
    }

    // Parse the response
    const data: GroqResponse = await response.json();
    
    // Return the generated summary
    return data.choices[0].message.content;
  } catch (error) {
    console.error('Error generating conversation summary:', error);
    throw error;
  }
} 