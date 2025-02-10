# cps_assistant_rag
This repository contains the code for implementing a RAG (Retrieval-Augmented Generation) system that stores information about various programs in the College of Professional Studies in a vector database. The system utilizes open-source large language models (LLMs) to act as a student assistant.

# How to Run the CPS AI Assistant? 

Setup the python environment:
•	Have a machine with Python 3.9 or greater installed.
•	Go to the GitHub link to download the zip or clone the repo      https://github.com/jibinb961/cps_assistant_rag
•	Open a terminal on the extracted folder
•	Use this comment to create a virtual environment “ python3 -m venv venv”
•	Use this comment to activate the environment “source venv/bin/activate”
•	Use this comment to install all the python requirements “pip install -r requirements.txt”

Setup .env file

•	Navigate to the cps_assistant_rag folder and open the folder in any code editor.
•	Edit file “. env” to add the new Groq Api key.
•	Go to groq website to create a new account https://groq.com
•	Navigate to the API keys on the left side to create a new API Key.
•	Update the variable in .env file to add GROQ_API_KEY=your_actual_api_key_here
•	Save the .env file.

Setup Ollama locally for running text embedding model

•	Go to Ollama website to create a new account. https://ollama.com
•	Download the Ollama to your local system 
•	Continue with the installation 
•	Open a new terminal and use the following code to install embedding model locally “ollama run nomic-embed-text”

Setup Streamlit UI 

•	Navigate to the dev folder under cps_assistant_rag folder 
•	Open a new terminal and run “streamlit run rag.py”
•	This will open a new browser at  http://localhost:8501 
•	Note: http://localhost:8501 is default address given by streamlit.

