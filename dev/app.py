import requests
from datetime import datetime, timezone
from urllib.parse import urlparse
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from supabase import create_client
import streamlit as st
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# ✅ Ensure correct Supabase key usage
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Ensure using SERVICE_KEY

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("❌ ERROR: Supabase URL and SERVICE KEY must be set in .env")

# ✅ Initialize Supabase client only once
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

class OllamaEmbeddings:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/embeddings"
        
    def embed_query(self, text):
        response = requests.post(
            self.endpoint,
            json={
                "model": "nomic-embed-text:latest",
                "prompt": text
            }
        )
        response_data = response.json()
        if "embedding" not in response_data:
            raise ValueError(f" ERROR: Failed to get embedding. Response: {response_data}")
        return response_data["embedding"]
    
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

# ✅ Initialize local embeddings
embeddings = OllamaEmbeddings()

# ✅ Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
)

# ✅ Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def extract_program_details(query):
    """
    Extract program mode and location with improved accuracy
    """
    extraction_prompt = """
    Analyze the query to extract:
    1. Program delivery mode (Online/On Campus)
    2. Campus location
    3. Any specific requirements mentioned
    
    Return as JSON with clear reasoning:
    {
        "program_mode": "string (Online/On Campus/any)",
        "campus_location": "string (specific location/any)",
        "requirements": ["list of specific requirements"],
        "confidence": "high/medium/low"
    }
    
    Consider contextual clues like:
    - Explicit mentions of online/campus
    - Location references
    - Time/schedule references
    
    Query: {query}
    """
    
    response = llm.invoke(extraction_prompt)
    try:
        json_response = json.loads(response.content)
        # Only use extracted details if confidence is high
        if json_response.get('confidence', 'low') == 'low':
            return {"program_mode": "any", "campus_location": "any"}
        return json_response
    except json.JSONDecodeError:
        return {"program_mode": "any", "campus_location": "any"}

def get_relevant_chunks(query, top_k=3):
    """
    Retrieve relevant chunks from Supabase with improved query analysis
    """
    # Extract program details with improved prompt
    program_details = extract_program_details(query)
    
    # Extract program name using LLM
    program_name = extract_program_name(query)
    
    # Generate query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Get relevant chunks with enhanced metadata matching
    response = supabase.rpc(
    'match_documents_with_metadata',
    {
        'query_embedding': query_embedding,
        'match_count': 3,
        'p_mode': program_details['program_mode'],
        'p_location': program_details['campus_location'],
        'query_text': query.lower(),
        'program_name': program_name
    }
    ).execute()
    print(program_details)
    
    results = response.data
    print(len(results))
    
    # Post-process results to ensure relevance
    filtered_results = filter_relevant_chunks(results, query, program_name)
    #print(filtered_results)
    
    return filtered_results

def filter_relevant_chunks(chunks, query, program_name):
    """
    Post-process chunks to ensure relevance
    """
    filtered_chunks = []
    query_terms = set(query.lower().split())
    
    for chunk in chunks:
        relevance_score = 0
        content = chunk.get('content', '').lower()
        title = chunk.get('title', '').lower()
        
        # Check if chunk is about the specific program
        if program_name != "any" and program_name in (title + " " + content):
            relevance_score += 2
            
        # Check for query term matches
        for term in query_terms:
            if term in content:
                relevance_score += 1
            if term in title:
                relevance_score += 2
                
        # Check for contextual relevance (e.g., if asking about cost, prioritize chunks with cost info)
        if any(term in query.lower() for term in ['cost', 'price', 'fee', 'tuition']):
            if any(term in content.lower() for term in ['cost', 'price', 'fee', 'tuition', '$']):
                relevance_score += 3
                
        if relevance_score > 2:  # Threshold for relevance
            chunk['relevance_score'] = relevance_score
            filtered_chunks.append(chunk)
    
    # Sort by relevance score
    return sorted(filtered_chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)

def extract_program_name(query):
    """
    Extract specific program name from the query using LLM
    """
    extraction_prompt = """
    Extract the specific program name from the query. If no program is mentioned, return "any".
    Return only the program name without any JSON formatting or additional text.
    
    Query: {query}
    """
    
    response = llm.invoke(extraction_prompt.format(query=query))
    return response.content.strip().lower()
    

def format_context(chunks):
    """
    Format retrieved chunks with improved context and metadata
    """
    if not chunks:
        return "No relevant program information found."
    
    # Group chunks by URL to maintain context
    url_groups = {}
    for chunk in chunks:
        url = chunk.get('url', 'N/A')
        if url not in url_groups:
            url_groups[url] = []
        url_groups[url].append(chunk)
    
    formatted_contexts = []
    for url, url_chunks in url_groups.items():
        # Sort chunks by chunk number to maintain proper order
        url_chunks.sort(key=lambda x: x.get('chunk_number', 0))
        
        # Extract page title and metadata
        page_title = url_chunks[0].get('title', 'N/A')
        metadata = url_chunks[0].get('metadata', {})
        
        # Format content with proper context
        content_parts = []
        for chunk in url_chunks:
            content = chunk.get('content', '').strip()
            if content:
                content_parts.append(content)
        
        formatted_context = f"""
        Source: {page_title}
        URL: {url}
        Program Mode: {metadata.get('program_mode', 'N/A')}
        Location: {metadata.get('campus_location', 'N/A')}
        
        Content:
        {' '.join(content_parts)}
        """
        formatted_contexts.append(formatted_context)
    
    return "\n\n---\n\n".join(formatted_contexts)

def generate_response(query, context, chat_history):
    """
    Generate response using Groq LLM with context and chat history
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about college programs. 
        Use the provided context to answer questions accurately. If the information isn't in 
        the context, say so. Consider the chat history for better context awareness.
        
        Context: {context}"""),
        ("human", "{question}")
    ])
    
    messages = prompt.format_messages(
        context=context,
        question=query
    )
    
    response = llm.invoke(messages)
    
    return response.content if isinstance(response, AIMessage) else "Error: Could not generate response."

# ✅ Streamlit UI
st.title("College Program Information Assistant")

# ✅ Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Chat input
if prompt := st.chat_input("Ask about our college programs"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ✅ Retrieve relevant chunks from Supabase
    relevant_chunks = get_relevant_chunks(prompt)
    context = format_context(relevant_chunks)
    
    # ✅ Generate response
    with st.chat_message("assistant"):
        response = generate_response(prompt, context, st.session_state.messages)
        st.markdown(response)
    
    # ✅ Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
