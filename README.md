# ResearchPaperQA – Academic Paper Q&A Chatbot with RAG

**Multilingual-capable RAG-powered chatbot that lets you upload research papers (PDF) and ask natural-language questions about their content — with cited answers and conversation memory.**


## Why this project?
Most basic RAG demos just do "upload PDF → ask question".  
This one goes further:

- Smart chunking + overlap for better context retrieval  
- Conversation memory (remembers previous questions)  
- Source citation (shows exact page/chunk used in the answer)  
- Clean FastAPI backend + Streamlit frontend  
- Evaluation metrics included (faithfulness, answer relevance)  
- Designed to handle academic papers (arXiv-style PDFs) — great for students, researchers, interviewers

Perfect demo for AI/ML interviews in 2026 — shows real RAG engineering without any model fine-tuning.

## Features
- Upload one or multiple research papers (PDF)  
- Ask follow-up questions with full conversation history  
- Answers include inline citations (page numbers or chunk references)  
- Uses high-quality multilingual embeddings (supports English + Indic abstracts if needed)  
- Local vector store (Chroma) – no cloud costs during development  
- Streamlit chat interface + FastAPI backend option  
- Basic evaluation script using ragas / manual test set

## Tech Stack
- **Backend / RAG**: LangChain, LangChain-OpenAI, Chroma  
- **LLM**: OpenAI GPT-4o-mini (cheap & strong) or any compatible model  
- **Embeddings**: OpenAI text-embedding-3-small (or free Hugging Face multilingual models)  
- **Frontend**: Streamlit (chat UI)  
- **Document handling**: PyPDFLoader  
- **Deployment ready**: Docker support (coming soon)

## Project Structure
├── app.py                  # Streamlit frontend + full RAG pipeline
├── api.py                  # Optional FastAPI backend version
├── evaluate.py             # Simple evaluation script
├── requirements.txt
├── .env.example
├── README.md
├── data/                   # Put sample papers here (not committed)
└── chroma_db/              # Local vector store (git ignore)
