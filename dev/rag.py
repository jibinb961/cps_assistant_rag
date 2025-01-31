import streamlit as st
from typing import Optional, List
import requests
from datetime import datetime, timezone
from urllib.parse import urlparse
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables and initialize clients
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("ERROR: Supabase URL and SERVICE KEY must be set in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
client = Groq()

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

def get_available_programs():
    """Fetch available programs from Supabase"""
    try:
        response = supabase.from_('site_pages')\
            .select('title')\
            .execute()
        
        # Extract unique program titles
        programs = set()
        for row in response.data:
            if row['title']:
                # Clean and standardize the title
                title = row['title'].strip()
                if title:  # Ensure non-empty after stripping
                    programs.add(title)
        
        return sorted(list(programs))  # Return sorted list of unique programs
    except Exception as e:
        st.error(f"Error fetching programs: {str(e)}")
        return []

def generate_prompt(query: str, context: str) -> str:
    """
    Generate a RAG-based prompt that includes the query and context.
    
    Args:
        query (str): User's search query
        context (str): Concatenated context from relevant document chunks
    
    Returns:
        str: Formatted prompt for the LLM
    """
    prompt_template = """ You are an AI assistant for the College of Professional Studies at Northeastern University, providing detailed and relevant information about course programs.
    If you cannot find the relevant information, ask the user to start a new search for course specific search results in this assistant.  
**Instructions:**  
- Use the provided context to answer the query clearly and concisely.  
- Format the response in markdown for readability.  
- Include useful URLs for further details.  
- If no relevant information is found from the context to user query, ask the user to Search in course specific section of this Assistant.
- If the context lacks sufficient details, state the limitation and suggest the user to search for course specific queries of this Assistant 

**Context:**  
{context}  

**User Query:**  
{query}  
    """
    
    return prompt_template.format(context=context, query=query)

def concatenate_chunks(chunks: List[dict], max_length: int = 100000) -> str:
    """
    Concatenate chunks of text, respecting a maximum length limit.
    
    Args:
        chunks (List[dict]): List of document chunks from search results
        max_length (int): Maximum total length of concatenated text
    
    Returns:
        str: Concatenated context from chunks
    """
    context_parts = []
    current_length = 0
    
    for chunk in chunks:
        chunk_text = f"[From {chunk.get('title', 'Unknown Source')}]\n{chunk.get('content', '')}\n\n"
        
        # Check if adding this chunk would exceed max_length
        if current_length + len(chunk_text) <= max_length:
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        else:
            break
    print("".join(context_parts))
    
    return "".join(context_parts)

def stream_groq_response(prompt: str) -> None:
    """
    Generate and stream a response from Groq using LangChain.
    
    Args:
        prompt (str): Formatted prompt for the LLM
    """
    try:
        # Initialize Groq model
        chat = ChatGroq(
            model="llama3-70b-8192",  # You can change this to other available models
            streaming=True,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Create an empty container for streaming
        response_container = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in chat.stream(prompt):
            full_response += chunk.content
            response_container.markdown(full_response)
        
        return full_response
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def process_search_results(query: str, results: List[dict]) -> None:
    """
    Process search results and generate a streaming response.
    
    Args:
        query (str): User's search query
        results (List[dict]): Search result chunks
    """
    if not results:
        st.warning("No results found for your query.")
        return
    
    # Concatenate context from chunks
    context = concatenate_chunks(results)
    
    # Generate prompt with context
    rag_prompt = generate_prompt(query, context)
    
    # Stream Groq response
    st.write("Generating response...")
    stream_groq_response(rag_prompt)

def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_program' not in st.session_state:
        st.session_state.selected_program = None
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = None
    if 'program_search' not in st.session_state:
        st.session_state.program_search = ""

def reset_session():
    """Reset session state"""
    st.session_state.selected_program = None
    st.session_state.search_mode = None
    st.session_state.program_search = ""

def filter_programs(programs, search_term):
    """Filter programs based on search term"""
    if not search_term:
        return programs
    search_term = search_term.lower()
    return [prog for prog in programs if search_term in prog.lower()]

def get_relevant_chunks(query, program_name: Optional[str] = None, top_k=10):
    """
    Modified version of get_relevant_chunks that includes program_name in the filter
    """
    try:
        # Generate query embedding
        embeddings = OllamaEmbeddings()
        query_embedding = embeddings.embed_query(query)
        
        # Prepare filter based on whether program_name is provided
        filter_params = {
            'source': 'cps_program_docs'
        }
        if program_name:
            filter_params['program_name'] = program_name  # Changed from program_name to title
        
        print(filter_params)
        
        # Get relevant chunks with filter
        response = supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': top_k,
                'search_mode': program_mode,
                'filter': filter_params
            }
        ).execute()
        #print(response.data)
        return response.data or []
        
    except Exception as e:
        st.error(f"Error in search: {str(e)}")
        return []
program_mode = 'general'

def main():
    st.title("AI Assistant for CPS Programs")
    
    # Initialize session state
    initialize_session_state()
    
    # Reset button
    if st.button("Start New Search"):
        reset_session()
    
    # If search mode not selected, show initial options
    if st.session_state.search_mode is None:
        st.write("What would you like to do?")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Search Specific Program"):
                st.session_state.search_mode = "specific"
                program_mode = 'specific'
        with col2:
            if st.button("General Search"):
                st.session_state.search_mode = "general"
                program_mode = 'general'
                st.session_state.selected_program = ""
    
    # If specific program mode selected but program not yet chosen
    if st.session_state.search_mode == "specific" and st.session_state.selected_program is None:
        # Get all available programs
        all_programs = get_available_programs()
        
        if all_programs:
            # Add search box for programs
            # st.text_input(
            #     "Search for a program:",
            #     key="program_search",
            #     value=st.session_state.program_search
            # )
            
            # Filter programs based on search
            filtered_programs = filter_programs(all_programs, st.session_state.program_search)
            
            # Show filtered programs in a selectbox
            if filtered_programs:
                selected = st.selectbox(
                    "Select a program:",
                    options=filtered_programs,
                    key="program_selector"
                )
                if st.button("Confirm Program"):
                    st.session_state.selected_program = selected
            else:
                st.warning("No programs match your search.")
        else:
            st.error("No programs available.")
    
    # Show search interface once mode and program (if applicable) are selected
    if st.session_state.search_mode is not None:
        # Show current mode and selected program
        st.write("---")
        st.write(f"Current Mode: {'General Search' if st.session_state.search_mode == 'general' else 'Program-Specific Search'}")
        if st.session_state.search_mode == "specific" and st.session_state.selected_program:
            st.write(f"Selected Program: {st.session_state.selected_program}")
        
        # Search interface
        query = st.text_input("Enter your search query:")
        if st.button("Search"):
            if query:
                # Get relevant chunks with the selected program (empty string for general search)
                program_name = st.session_state.selected_program if st.session_state.search_mode == "specific" else ""
                results = get_relevant_chunks(query, program_name=program_name)
                
                # Process and display results using the new function
                process_search_results(query, results)
            else:
                st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()