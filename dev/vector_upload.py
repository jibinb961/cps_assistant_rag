import os
import sys
import json
import asyncio
import requests
import httpx
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import cohere
import random
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VectorDBUploader')

@dataclass
class ProgramInfo:
    url: str
    title: str
    summary: str
    content: str
    chunk_number: int
    embedding: List[float]
    metadata: Dict[str, Any]

class VectorDBUploader:
    def __init__(self, 
                 url: str, 
                 cohere_api_key: str,
                 output_file: str = "program_data.jsonl",
                 info_type: str = "CPS Programs", 
                 source: str = "cps_program_docs",
                 format_content: bool = True,
                 cookies: Dict[str, str] = None):
        """
        Standalone uploader that uses Ollama for summarization and Cohere for embeddings.
        
        Args:
            url (str): The URL to process
            cohere_api_key (str): Cohere API key for embeddings
            output_file (str): File to save the extracted data
            info_type (str): Type of information ("CPS Programs", "Coop Information", or "Others")
            source (str): Source identifier for the content
            format_content (bool): Whether to format the content
            cookies (Dict[str, str]): Optional cookies for authentication
        """
        self.url = url
        self.cohere_client = cohere.Client(cohere_api_key)
        self.output_file = output_file
        self.info_type = info_type
        self.source = source
        self.format_content = format_content
        self.cookies = cookies
        self.session = requests.Session()  # Use a session for persistent cookies
        
        # Initialize supabase client if credentials are available
        self.supabase = None
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        if supabase_url and supabase_key:
            try:
                from supabase import create_client
                self.supabase = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized successfully")
            except ImportError:
                logger.warning("Supabase package not installed. Database uploads disabled.")
            except Exception as e:
                logger.error(f"Error initializing Supabase client: {e}")

    async def insert_chunk(self, chunk: ProgramInfo):
        """Insert a processed chunk into Supabase if available."""
        if not self.supabase:
            logger.info(f"Supabase not configured. Skipping database insert for chunk {chunk.chunk_number}")
            return None
            
        try:
            data = {
                "url": chunk.url,
                "chunk_number": chunk.chunk_number,
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding
            }
            
            result = self.supabase.table("site_pages").insert(data).execute()
            logger.info(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
            return result
        except Exception as e:
            logger.error(f"Error inserting chunk: {e}")
            return None

    def process_and_modify_markdown(self, content: str) -> str:
        """Clean and format markdown content by removing unwanted sections and formatting."""
        # Split the content into lines and work from line 80 onward (specific to CPS content)
        lines = content.split("\n")
        relevant_lines = lines[80:] if len(lines) > 80 else lines  # Skip header section if long enough

        filtered_lines = []
        keep_paragraph = False
        special_chars = set(['@', '%', '&', '+', '=', '|', '<', '>', '^', '~', '-'])

        for index, line in enumerate(relevant_lines):
            stripped_line = line.strip()

            # Stop processing if we reach the unwanted section
            if stripped_line.startswith("Our enrollment representatives"):
                break

            # Skip specific index range (lines 83 to 106)
            if 83 <= (index + 81) <= 106 and len(lines) > 106:
                continue

            # Skip lines starting with special characters
            if stripped_line and stripped_line[0] in special_chars:
                continue

            # Skip image markdown
            if stripped_line.startswith("![]"):
                continue

            # Skip additional hyperlinks
            if stripped_line.startswith("* ["):
                continue

            # Fix link formatting by keeping only the URL inside angle brackets
            line = re.sub(r"\[([^\]]+)\]\([^\(\)<>]*<([^<>]+)>\)", r"[\1](\2)", line)

            # Preserve and clean markdown headings
            if stripped_line.startswith("#"):
                line = re.sub(r'#+\s+', lambda m: m.group().strip() + ' ', line)
                filtered_lines.append(line)
                keep_paragraph = True

            # Preserve bullet points
            elif stripped_line.startswith("*") and len(stripped_line) > 1:
                line = re.sub(r'\*\s*', '* ', stripped_line)
                filtered_lines.append(line)
                keep_paragraph = True

            # Handle paragraph content
            elif keep_paragraph and stripped_line:
                line = re.sub(r'\\([\\`*_{}[\]()#+.!-])', r'\1', line)  # Remove escape chars
                filtered_lines.append(line)

            # Handle URLs
            elif "http" in stripped_line and not stripped_line.startswith("["):
                filtered_lines.append(line)

            # Handle empty lines
            elif stripped_line == "":
                filtered_lines.append("")
                keep_paragraph = False

        # Join lines and clean up multiple empty lines
        formatted_content = "\n".join(filtered_lines)
        formatted_content = re.sub(r'\n{3,}', '\n\n', formatted_content)

        # Modify the section starting with "## Take a Quick Look"
        pattern = r'## Take a Quick Look(.*?)(?=\n\n|$)'
        match = re.search(pattern, formatted_content, re.DOTALL)
        
        if match:
            # Extract the section content and remove extra spaces and newlines
            section_content = match.group(1).strip()
            
            # Separate lines by newlines
            lines = section_content.split('\n')
            
            # The first line is a description, so we leave it as is
            modified_section = lines[0] + "\n" if lines else ""
            
            # Iterate through the following lines, which should be in pairs (label, value)
            for i in range(1, len(lines) - 1, 2):
                label = lines[i].strip()
                value = lines[i + 1].strip() if i + 1 < len(lines) else ''
                
                # Format them as "Label : Value"
                if label and value:
                    modified_section += f"{label} : {value}\n"
            
            # Replace the original section with the newly formatted one
            formatted_content = formatted_content.replace(match.group(0), f"## Take a Quick Look\n{modified_section}")

        return formatted_content.strip()
    
    def chunk_program_content(self, text: str) -> List[str]:
        """
        Split text content into three chunks based on specific markers:
        1. Content before 'course plan'
        2. Content between 'course plan' and '## Cost and Tuition'
        3. Content after '## Cost and Tuition'
        """
        # Split the text into lines while preserving empty lines
        lines = text.splitlines(keepends=True)
        
        chunk1_lines = []
        chunk2_lines = []
        chunk3_lines = []
        
        # Track which chunk we're currently building
        current_chunk = 1
        
        # Process each line
        for line in lines:
            # Check for first split condition (course plan)
            if current_chunk == 1 and 'course plan' in line.lower():
                current_chunk = 2
                chunk2_lines.append(line)  # Include the splitting line in chunk 2
                continue
                
            # Check for second split condition (Cost and Tuition)
            if current_chunk == 2 and line.strip().startswith("## Cost and Tuition"):
                current_chunk = 3
                chunk3_lines.append(line)  # Include the splitting line in chunk 3
                continue
                
            # Add line to appropriate chunk
            if current_chunk == 1:
                chunk1_lines.append(line)
            elif current_chunk == 2:
                chunk2_lines.append(line)
            else:
                chunk3_lines.append(line)
        
        # Join the lines back together for each chunk
        chunk1 = ''.join(chunk1_lines)
        chunk2 = ''.join(chunk2_lines)
        chunk3 = ''.join(chunk3_lines)
        
        # If no splits were found, handle appropriately
        if not chunk2_lines and not chunk3_lines:
            if chunk1:
                # If there's only one chunk but it's large, split it roughly into thirds
                if len(chunk1) > 3000:
                    third = len(chunk1.splitlines()) // 3
                    lines = chunk1.splitlines(keepends=True)
                    return [''.join(lines[:third]), ''.join(lines[third:2*third]), ''.join(lines[2*third:])]
                return [chunk1, '', '']
            return ['', '', '']
        elif not chunk3_lines:
            return [chunk1, chunk2, '']
        
        return [chunk1, chunk2, chunk3]

    async def get_ollama_embedding(self, text: str) -> List[float]:
        """Get embedding vector from local Ollama model."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={
                        "model": "nomic-embed-text:latest",
                        "prompt": text
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                if not result.get("embedding"):
                    logger.warning(f"Warning: Empty embedding returned from Ollama. API response: {result}")
                    return None
                
                embedding = result["embedding"]
                logger.info(f"Successfully generated Ollama embedding with dimension: {len(embedding)}")
                
                return embedding
        except Exception as e:
            logger.error(f"Error getting Ollama embedding: {e}")
            return None

    def get_cohere_embedding(self, text: str) -> List[float]:
        """Get embedding vector from Cohere API."""
        try:
            response = self.cohere_client.embed(
                texts=[text],
                model='embed-english-v3.0',
                input_type="search_document"
            )
            
            if not response.embeddings or len(response.embeddings) == 0:
                logger.warning("Warning: Empty embedding returned from Cohere API")
                return None
            
            embedding = response.embeddings[0]
            logger.info(f"Successfully generated Cohere embedding with dimension: {len(embedding)}")
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting Cohere embedding: {e}")
            return None

    async def extract_program_info(self, chunk: str, url: str, chunk_number: int, program_details: dict = None) -> Optional[ProgramInfo]:
        """
        Extract program information using local Ollama LLM for summary and Cohere for embeddings.
        """
        system_prompt = """You are a JSON-only response API that extracts program information from educational content.
        Your task is to extract a brief summary.
        RESPONSE FORMAT:
        You must return a valid JSON object with exactly this structure:
        {
            "summary": "Program summary here"
        }
        RULES:
        1. ONLY return the JSON object, no other text
        4. Summary should be 2-3 sentences describing what the program and the specific section is about.
        5. If you can't find the information, take the first few lines of the content.
        6. Never include markdown, HTML, or special characters in the values
        7. Always maintain valid JSON syntax
        8. Never include explanations or notes outside the JSON object"""
        
        try:
            # Extract program details from first chunk only
            if chunk_number == 1:
                first_lines = chunk.strip().split('\n')
                if first_lines:
                    program_line = first_lines[0].strip('# ').strip()
                    og_pg_name = first_lines[0].strip('# ').strip()
                    
                    # Normalize all dashes (en dash and em dash) to a standard hyphen (-)
                    program_line = program_line.replace("\u2013", "-").replace("\u2014", "-")
                    
                    # Split using the standard hyphen, ensuring only two parts
                    program_parts = program_line.split(" - ", 1)  # Split at the first occurrence
                    
                    program_name = program_parts[0].strip()
                    program_mode = program_parts[1].strip() if len(program_parts) > 1 else ''
                    
                    # Determine program mode and campus location
                    if 'Online' in program_mode:
                        mode = 'online'
                        campus_location = 'online'
                    else:
                        mode = 'on campus'
                        campus_location = program_mode  # If empty, campus_location will be ''
                    if not campus_location:
                        campus_location = 'Boston'  # Default

                    # Create program details dictionary
                    program_details = {
                        'program_name': og_pg_name.replace("\u2013", "-").replace("\u2014", "-"),
                        'program_mode': mode,
                        'campus_location': campus_location
                    }
            
            # Get summary using Ollama
            summary = ""
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3.2:3b",
                            "prompt": f"{system_prompt}\n\nURL: {url}\n\nContent:\n{chunk[:1000]}",
                            "stream": False
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Check if 'response' key exists and is not empty
                    if "response" not in result or not result["response"].strip():
                        logger.warning(f"Warning: Empty response from LLM for {url}")
                        summary = "\n".join(chunk.splitlines()[:6])  # Fallback: First few lines
                    else:
                        raw_text = result["response"]

                        # Attempt JSON parsing
                        try:
                            extracted = json.loads(raw_text)
                            summary = extracted.get("summary", "")
                        except json.JSONDecodeError:
                            logger.warning(f"Warning: Invalid JSON received for {url}. Attempting fallback extraction.")
                            
                            # Try extracting a valid JSON object from the raw text using regex
                            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                            if json_match:
                                try:
                                    extracted = json.loads(json_match.group(0))
                                    summary = extracted.get("summary", "")
                                except json.JSONDecodeError:
                                    summary = "\n".join(chunk.splitlines()[:3])  # Fallback: First few lines
                            else:
                                summary = "\n".join(chunk.splitlines()[:3])  # Fallback: First few lines
            except Exception as e:
                logger.error(f"Error getting summary: {e}")
                summary = "\n".join(chunk.splitlines()[:3])  # Fallback: First few lines
                
            # Generate embeddings using Cohere
            embedding = self.get_cohere_embedding(chunk)
            
            if not embedding:
                logger.error(f"Failed to generate embedding for {url} chunk {chunk_number}")
                return None
            
            # Create metadata dictionary
            metadata = {
                "source": self.source,
                "url": url,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "chunk_type": f"section_{chunk_number}"
            }
            
            # Add program details to metadata if available
            if program_details:
                for key, value in program_details.items():
                    metadata[key] = value
            
            # Create ProgramInfo object
            program_info = ProgramInfo(
                url=url,
                title=program_details.get('program_name', url) if program_details else url,
                summary=summary,
                content=chunk,
                chunk_number=chunk_number,
                embedding=embedding,
                metadata=metadata
            )
            
            return program_info
            
        except Exception as e:
            logger.error(f"Error extracting program info: {e}")
            return None
    
    async def save_to_file(self, program_info_list: List[ProgramInfo]):
        """Save program data to a JSONL file."""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for program_info in program_info_list:
                    data = {
                        "url": program_info.url,
                        "title": program_info.title,
                        "summary": program_info.summary,
                        "content": program_info.content,
                        "chunk_number": program_info.chunk_number,
                        "embedding": program_info.embedding,
                        "metadata": program_info.metadata
                    }
                    f.write(json.dumps(data) + '\n')
            
            logger.info(f"Successfully saved {len(program_info_list)} chunks to {self.output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return False
    
    async def process_url(self):
        """Process the URL to extract program information."""
        try:
            # Set headers to mimic a real browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                "Referer": "https://www.google.com/"  # Add a referer to make it look like the request came from Google
            }
            
            # Add cookies if provided
            if self.cookies:
                for name, value in self.cookies.items():
                    self.session.cookies.set(name, value)
            
            # Fetch the webpage content with proper headers
            logger.info(f"Fetching content from {self.url}")
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    response = self.session.get(self.url, headers=headers, timeout=30)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise
            
            # Extract content from the response
            content = response.text
            
            # Process the content
            if self.format_content:
                content = self.process_and_modify_markdown(content)
            
            # Split content into chunks
            chunks = self.chunk_program_content(content)
            
            # Process each chunk
            program_info_list = []
            program_details = None
            
            for i, chunk in enumerate(chunks, 1):
                if not chunk.strip():
                    logger.info(f"Skipping empty chunk {i} for {self.url}")
                    continue
                
                logger.info(f"Processing chunk {i} for {self.url}")
                program_info = await self.extract_program_info(chunk, self.url, i, program_details)
                
                if not program_info:
                    logger.warning(f"Failed to extract info for chunk {i}")
                    continue
                
                # Save the program details from the first chunk for later use
                if i == 1 and program_info.metadata.get('program_name'):
                    program_details = {
                        'program_name': program_info.metadata.get('program_name'),
                        'program_mode': program_info.metadata.get('program_mode'),
                        'campus_location': program_info.metadata.get('campus_location')
                    }
                
                program_info_list.append(program_info)
                
                # Insert into database if available
                await self.insert_chunk(program_info)
            
            # Save to file
            await self.save_to_file(program_info_list)
            
            logger.info(f"Successfully processed {self.url}")
            return program_info_list
            
        except Exception as e:
            logger.error(f"Error processing URL {self.url}: {e}")
            return None
    
    @classmethod
    async def process_urls_from_file(cls, 
                                    file_path: str, 
                                    cohere_api_key: str,
                                    output_file: str = "program_data.jsonl",
                                    source: str = "cps_program_docs",
                                    cookies: Dict[str, str] = None):
        """Process multiple URLs from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            results = []
            for url in urls:
                logger.info(f"Processing URL: {url}")
                uploader = cls(url, cohere_api_key, output_file, source=source, cookies=cookies)
                result = await uploader.process_url()
                if result:
                    results.extend(result)
                
                # Add a delay between requests to avoid hitting rate limits
                delay = 3 + random.uniform(1, 3)  # Random delay between 4-6 seconds
                logger.info(f"Sleeping for {delay:.2f} seconds before next request")
                await asyncio.sleep(delay)
            
            logger.info(f"Successfully processed {len(urls)} URLs with {len(results)} total chunks")
            return results
        except Exception as e:
            logger.error(f"Error processing URLs from file: {e}")
            return None

    @staticmethod
    def load_cookies(cookie_file: str) -> Dict[str, str]:
        """Load cookies from a JSON file."""
        if not cookie_file or not os.path.exists(cookie_file):
            return None
            
        try:
            with open(cookie_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cookies: {e}")
            return None

async def main():
    """Main function to run the script."""
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Process URLs to create vector chunks")
    parser.add_argument("--url", help="URL to process")
    parser.add_argument("--url-file", help="File containing URLs to process")
    parser.add_argument("--output", default="program_data.jsonl", help="Output file for the chunks")
    parser.add_argument("--source", default="cps_program_docs", help="Source tag for the chunks")
    parser.add_argument("--cookie-file", help="JSON file containing cookies for authentication")
    
    args = parser.parse_args()
    
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        logger.error("COHERE_API_KEY environment variable not set")
        return

    # Load cookies if provided
    cookies = VectorDBUploader.load_cookies(args.cookie_file)
    if args.cookie_file and not cookies:
        logger.warning("Cookie file specified but couldn't be loaded")

    if args.url:
        uploader = VectorDBUploader(args.url, cohere_api_key, args.output, source=args.source, cookies=cookies)
        await uploader.process_url()
    elif args.url_file:
        await VectorDBUploader.process_urls_from_file(
            file_path=args.url_file,
            cohere_api_key=cohere_api_key,
            output_file=args.output,
            source=args.source,
            cookies=cookies
        )
    else:
        logger.error("Either --url or --url-file must be provided")
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    asyncio.run(main())
