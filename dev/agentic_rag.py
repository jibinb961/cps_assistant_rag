import streamlit as st
from typing import Optional, List, Dict, Any, Tuple
import requests
from datetime import datetime, timezone
from urllib.parse import urlparse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv
import os
import re
import json
from PIL import Image

# Load environment variables and initialize clients
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("ERROR: Supabase URL and SERVICE KEY must be set in .env")

if not GROQ_API_KEY:
    raise ValueError("ERROR: GROQ API KEY must be set in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
client = Groq(api_key=GROQ_API_KEY)

# Configure Streamlit page
st.set_page_config(
    page_title="CPS AI Assistant",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI Assistant for Northeastern University's College of Professional Studies"
    }
)

# Apply custom styles
st.markdown("""
    <style>
    /* Main styles */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Title styling */
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
        color: white;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 16px;
        color: #FAFAFA;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: transparent;
        color: white;
        border: 1px solid white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    /* Button hover effect */
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    /* Text color */
    .stMarkdown {
        color: white;
    }
    
    /* Chat container */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .chat-message.user {
        background-color: rgba(120, 120, 120, 0.1);
    }
    
    .chat-message.bot {
        background-color: rgba(255, 65, 0, 0.05);
    }
    
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
    }
    
    .chat-message .message {
        flex-grow: 1;
        padding: 0;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

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
            raise ValueError(f"ERROR: Failed to get embedding. Response: {response_data}")
        return response_data["embedding"]
    
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

def get_available_programs():
    """Fetch available programs from Supabase"""
    try:
        response = supabase.from_('site_pages')\
            .select('title')\
            .execute()
        
        programs = set()
        for row in response.data:
            if row['title']:
                title = row['title'].strip()
                if title:
                    programs.add(title)
        
        return sorted(list(programs))
    except Exception as e:
        return []

def generate_prompt(query: str, context: str) -> str:
    """Generate a prompt for the AI model based on user query and context"""
    prompt_template = """ You are an AI assistant for the College of Professional Studies at Northeastern University, providing detailed and relevant information about course programs.
    If you cannot find the relevant information, just state so politely.  
**Instructions:**  
- Use the provided context to answer the query clearly and concisely.  
- Format the response in markdown for readability.  
- Include useful URLs for further details.  
- If no relevant information is found from the context to user query, state that clearly.
- If the context lacks sufficient details, state the limitation.

**Context:**  
{context}  

**User Query:**  
{query}  
    """
    
    return prompt_template.format(context=context, query=query)

def concatenate_chunks(chunks: List[dict], max_length: int = 100000) -> str:
    """Concatenate text chunks into a single string with a maximum length"""
    context_parts = []
    current_length = 0
    
    for chunk in chunks:
        chunk_text = f"[From {chunk.get('title', 'Unknown Source')}]\n{chunk.get('content', '')}\n\n"
        
        if current_length + len(chunk_text) <= max_length:
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        else:
            break
    
    return "".join(context_parts)

def get_relevant_chunks(query, program_name: Optional[str] = None, top_k=10):
    """Fetch relevant chunks based on the user query and program name"""
    try:
        embeddings = OllamaEmbeddings()
        query_embedding = embeddings.embed_query(query)
        
        filter_params = {
            'source': 'coop_information' if program_name == "coop_information" else 'cps_program_docs'
        }
        
        if program_name != "coop_information" and program_name != "":
            filter_params['program_name'] = program_name
        
        search_mode = 'coop' if program_name == "coop_information" else 'general' if program_name == "" else 'specific'
        
        response = supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': top_k,
                'search_mode': search_mode,
                'filter': filter_params
            }
        ).execute()
        
        return response.data or []
        
    except Exception as e:
        return []

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

# Define tool schemas using Pydantic
class ProgramSpecificSearchInput(BaseModel):
    program: str = Field(description="The specific program to search information about")
    query: str = Field(description="The query about the specific program")

class GeneralSearchInput(BaseModel):
    query: str = Field(description="The general query about any program or multiple programs")

class CoopSearchInput(BaseModel):
    query: str = Field(description="The query about coop programs or opportunities")

# Define tool functions
def program_specific_search(input_data: Any) -> str:
    """
    Search for information about a specific program. Use this when the user is asking about 
    details of a particular program like courses, requirements, structure, etc.
    """
    # Handle different input types
    if isinstance(input_data, str):
        try:
            # Try to parse as JSON
            data = json.loads(input_data)
            program = data.get("program", "")
            query = data.get("query", input_data)
        except json.JSONDecodeError:
            # If not valid JSON, use a default program and use the whole string as query
            program = extract_program_names(input_data)[0] if extract_program_names(input_data) else ""
            query = input_data
    elif isinstance(input_data, list):
        # Handle array input - first element is program, second is query
        if len(input_data) >= 2:
            program = input_data[0]
            query = input_data[1]
        else:
            program = ""
            query = input_data[0] if input_data else ""
    else:
        # Handle object input
        try:
            program = input_data.program
            query = input_data.query
        except AttributeError:
            # Fallback if attributes don't exist
            program = ""
            query = str(input_data)
    
    results = get_relevant_chunks(query, program_name=program)
    
    if not results:
        return f"I couldn't find specific information about {program} related to your query. Please try a different query or program."
    
    context = concatenate_chunks(results)
    prompt = generate_prompt(query, context)
    
    # Use Groq for response generation
    chat = ChatGroq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY
    )
    
    response = chat.invoke(prompt)
    return response.content

def general_search(input_data: Any) -> str:
    """
    Search across all programs. Use this when the user is asking general questions 
    or wants to compare multiple programs.
    """
    # Handle different input types
    if isinstance(input_data, str):
        try:
            # Try to parse as JSON
            data = json.loads(input_data)
            query = data.get("query", input_data)
        except json.JSONDecodeError:
            # If not valid JSON, use the whole string as query
            query = input_data
    elif isinstance(input_data, list):
        # Handle array input
        query = input_data[0] if input_data else ""
    else:
        # Handle object input
        try:
            query = input_data.query
        except AttributeError:
            # Fallback if attributes don't exist
            query = str(input_data)
    
    results = get_relevant_chunks(query, program_name="")
    
    if not results:
        return "I couldn't find information related to your general query. Please try a more specific question."
    
    context = concatenate_chunks(results)
    prompt = generate_prompt(query, context)
    
    # Use Groq for response generation
    chat = ChatGroq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY
    )
    
    response = chat.invoke(prompt)
    return response.content

def coop_search(input_data: Any) -> str:
    """
    Search for coop related information. Use this when the user is asking about 
    coop programs, opportunities, requirements, etc.
    """
    # Handle different input types
    if isinstance(input_data, str):
        try:
            # Try to parse as JSON
            data = json.loads(input_data)
            query = data.get("query", input_data)
        except json.JSONDecodeError:
            # If not valid JSON, use the whole string as query
            query = input_data
    elif isinstance(input_data, list):
        # Handle array input
        query = input_data[0] if input_data else ""
    else:
        # Handle object input
        try:
            query = input_data.query
        except AttributeError:
            # Fallback if attributes don't exist
            query = str(input_data)
    
    results = get_relevant_chunks(query, program_name="coop_information")
    
    if not results:
        return "I couldn't find information related to your coop query. Please try a different question about coop programs."
    
    context = concatenate_chunks(results)
    prompt = generate_prompt(query, context)
    
    # Use Groq for response generation
    chat = ChatGroq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY
    )
    
    response = chat.invoke(prompt)
    return response.content

def create_agent():
    """Create a LangChain agent with tools for different search modes"""
    
    # Define tools
    tools = [
        Tool(
            name="program_specific_search",
            description="Search for information about a specific program. Use this when the user is asking about details of a particular program.",
            func=program_specific_search,
            args_schema=ProgramSpecificSearchInput
        ),
        Tool(
            name="general_search",
            description="Search across all programs. Use this when the user is asking general questions or wants to compare multiple programs.",
            func=general_search,
            args_schema=GeneralSearchInput
        ),
        Tool(
            name="coop_search",
            description="Search for coop related information. Use this when the user is asking about coop programs, co-op advisors, opportunities, requirements, etc.",
            func=coop_search,
            args_schema=CoopSearchInput
        )
    ]
    
    # Create a memory object with newer API
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the LLM
    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY
    )
    
    # Create the agent with tool-calling capabilities
    system_message = """You are an AI assistant for the College of Professional Studies at Northeastern University.
    Your goal is to provide helpful information about various programs, courses, and coop opportunities.
    
    When responding to users:
    1. For questions about specific programs, use the program_specific_search tool
    2. For general questions or comparisons between programs, use the general_search tool
    3. For questions about coop programs, co-op advisors, co-op opportunities, co-op requirements, etc., ALWAYS use the coop_search tool
    
    When using the program_specific_search tool, make sure to identify the specific program the user is asking about.
    Common programs include: 
    - "Analytics"
    - "Computer Science"
    - "Information Systems"
    - "Project Management"
    - And other graduate and undergraduate programs at Northeastern University's CPS
    
    IMPORTANT: Any questions related to co-op, including co-op advisors, co-op process, or anything with the word "coop" or "co-op" should ALWAYS use the coop_search tool.
    
    Always be helpful, concise, and accurate in your responses.
    Format your responses using markdown for better readability.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )
    
    return agent_executor

def format_message(message, is_user):
    """Format messages in a styled chat interface"""
    if is_user:
        avatar_url = "https://i.imgur.com/N7YOtz0.png"  # User avatar placeholder URL
        return f"""
        <div class="chat-message user">
            <img src="{avatar_url}" alt="User Avatar" class="avatar">
            <div class="message">{message}</div>
        </div>
        """
    else:
        avatar_url = "images/CPSBOT10.png"  # Bot avatar placeholder URL
        return f"""
        <div class="chat-message bot">
            <img src="{avatar_url}" alt="Bot Avatar" class="avatar">
            <div class="message">{message}</div>
        </div>
        """

def display_chat():
    """Display the chat interface with message history"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(format_message(message["content"], True), unsafe_allow_html=True)
        else:
            st.markdown(format_message(message["content"], False), unsafe_allow_html=True)

def extract_program_names(query):
    """Extract potential program names from a query"""
    all_programs = get_available_programs()
    found_programs = []
    
    for program in all_programs:
        if program.lower() in query.lower():
            found_programs.append(program)
    
    return found_programs

def main():
    """Main function to run the Streamlit app"""
    
    # Create two columns for title and image
    col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed

    with col1:
        # Title section with custom styling
        st.markdown('<p class="big-font">🐾 AI Assistant for CPS Programs</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="subtitle">Hey Huskies! 👋 Get instant answers about CPS programs, courses, and requirements.</p>', 
            unsafe_allow_html=True
        )

    with col2:
        # Display the image on the right
        image_path = "images/CPSBOT10.png"  # Path to your image file
        try:
            image = Image.open(image_path)  # Open the image
            st.image(image, width=150)  # Set width to 150 pixels
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

    # Initialize session state
    initialize_session_state()
    
    # Create agent if not in session state
    if 'agent' not in st.session_state:
        st.session_state.agent = create_agent()
    
    # Display chat history
    display_chat()
    
    # Chat input
    with st.container():
        # Add a separator
        st.write("---")
        
        # Create the chat input
        user_input = st.chat_input("💭 What would you like to know about CPS programs?")
        
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display the updated chat (with the new user message)
            with st.spinner("🤔 Thinking..."):
                try:
                    # Look for coop-related keywords for direct routing
                    if "coop" in user_input.lower() or "co-op" in user_input.lower() or "advisor" in user_input.lower():
                        # Directly use coop search for coop-related queries
                        agent_response = coop_search(user_input)
                    else:
                        # Get response from agent
                        response = st.session_state.agent.invoke({
                            "input": user_input,
                            "chat_history": st.session_state.chat_history
                        })
                        
                        # Extract the agent's response
                        agent_response = response["output"]
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    # Fallback to general search
                    agent_response = general_search(user_input)
                
                # Add AI message to chat history
                st.session_state.messages.append({"role": "assistant", "content": agent_response})
                
                # Update the langchain conversation history format
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=agent_response))
                
                # Rerun to update the UI
                st.rerun()

if __name__ == "__main__":
    main() 