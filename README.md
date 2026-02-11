

Live Demo: https://huggingface.co/spaces/zsezim/ardenture-rag
Demo is private to limit API usage. 
Please request access on HuggingFace if you'd like to use it!

I made a production-style Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs and ask questions grounded in the document content. The RAG performs semantic retrieval over document chunks and generates answers using an LLM hosted via Together AI. The UI is built with Gradio and deployed on Hugging Face Spaces.

## Tech Stack:
- Language: python 3.11
- Framework: LangChain
- Vector store: Chroma
- Embeddings: sentence-transformers
- LLM API: Together AI
- Frontend/UI: Gradio
- Hosting/Deployment: HuggingFace Spaces
- PDF parsing: pypdf

## Project structure:
```
rag-assistant/
├── app.py        
├── src/
│   └── rag_assistant/
│       ├── ingest.py 
│       ├── retriever.py  
│       ├── generator.py  
│       ├── rag.py        
│       └── settings.py   
├── requirements.txt
├── README.md
└── .gitignore
```

## Example use cases:
Financial reports querying
Legal documents semantic assistant
Internal documentation search
Client-facing document assistants
Other confidential docs

## Local setup:
1. Clone the repo

```
git clone https://github.com/zsezim/rag-assistant.git
cd rag-assistant
```
2. Create a virtual environment and activate it (warning: you must use python 3.11 or older!):
```
python3.11 -m venv venv
source venv/bin/activate
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Set environment variables (I used llama for my model, but you can pick a different one if you wish):
```
TOGETHER_API_KEY=your_together_api_key_here
TOGETHER_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
```
5. Run locally:
```
python app.py
```
## Future improvements:
Multi-document ingestion per session
Metadata filtering UI (page number, section)
Authentication (per-user storage)

## Author:
Sezim Zamirbekova Kaplan

Technical Data Scientist
GitHub: https://github.com/zsezim
Hugging Face: https://huggingface.co/zsezim