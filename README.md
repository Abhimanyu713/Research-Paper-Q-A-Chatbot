# 📄 Research Paper Q&A Chatbot

**A production-quality Retrieval Augmented Generation (RAG) system for academic paper question answering.**

Built as a portfolio project demonstrating end-to-end AI/ML engineering: document processing, embedding pipelines, vector databases, LLM orchestration, and evaluation.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                                                             │
│  PDF Upload → PyPDFLoader → RecursiveTextSplitter → Chunks  │
│                                      ↓                      │
│                          OpenAI Embeddings (3-small)         │
│                                      ↓                      │
│                           ChromaDB (local persist)           │
└─────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                             │
│  User Question                                              │
│       ↓                                                     │
│  ConversationBufferMemory (condense follow-ups)              │
│       ↓                                                     │
│  Similarity Search → Top-5 Chunks (ChromaDB)                │
│       ↓                                                     │
│  Prompt: [System] + [Context] + [History] + [Question]      │
│       ↓                                                     │
│  GPT-4o-mini → Grounded Answer + Citations                  │
│       ↓                                                     │
│  Streamlit UI / FastAPI Response                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
project_root/
├── app.py                  # Entry point (Streamlit / FastAPI)
├── api.py                  # FastAPI REST backend
├── evaluate.py             # Evaluation script (RAGAS / simple)
│
├── rag/
│   ├── loader.py           # PDF loading + metadata enrichment
│   ├── chunker.py          # Smart recursive text splitting
│   ├── embeddings.py       # OpenAI / HuggingFace embedding factory
│   ├── vector_store.py     # ChromaDB management
│   ├── retriever.py        # Semantic + hybrid retrieval
│   └── rag_chain.py        # ConversationalRetrievalChain + prompts
│
├── ui/
│   └── streamlit_app.py    # Chat interface
│
├── data/
│   └── chroma_db/          # Persisted ChromaDB (auto-created)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd research-paper-qa
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
# Open http://localhost:8501
```

### 4. (Optional) Run the FastAPI backend

```bash
uvicorn api:app --reload --port 8000
# Swagger docs at http://localhost:8000/docs
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `huggingface` |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `research_papers` | Collection name |

---

## 💡 Example Queries

After uploading a paper, try:

```
"What is the main contribution of this paper?"
"How does this compare to BERT/GPT baselines?"
"What datasets were used for evaluation?"
"Explain the methodology in simple terms."
"What were the key experimental results?"
"What limitations do the authors acknowledge?"
"What future work is suggested?"
```

**Follow-up questions (conversation memory):**
```
User: "What method did they use?"
AI:   "The authors used [X]... (Source: paper.pdf, Page 3)"

User: "What about the ablation study?"
AI:   "The ablation study on Page 7 shows..." ← memory in action
```

---

## 🔌 API Reference

### POST /upload
```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@attention_is_all_you_need.pdf"
```

### POST /ask
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the transformer architecture?"}'
```

Response:
```json
{
  "question": "What is the transformer architecture?",
  "answer": "The transformer uses self-attention mechanisms... (Source: paper.pdf, Page 2)",
  "sources": [
    {
      "file_name": "paper.pdf",
      "page": 2,
      "chunk_id": "paper.pdf_p2_c1",
      "snippet": "We propose a new simple network architecture..."
    }
  ]
}
```

### GET /history
```bash
curl http://localhost:8000/history
```

---

## 📊 Evaluation

### Simple evaluation (no ground truth needed)
```bash
python evaluate.py --mode simple --pdf path/to/paper.pdf
```

### Full RAGAS evaluation
```bash
python evaluate.py --mode ragas --pdf path/to/paper.pdf --output results.json
```

**Metric interpretation:**

| Metric | Good Score | Meaning |
|--------|-----------|---------|
| Faithfulness | > 0.8 | Low hallucination risk |
| Answer Relevancy | > 0.8 | Answers stay on-topic |
| Context Precision | > 0.7 | Retrieval mostly relevant |
| Context Recall | > 0.7 | Most needed chunks retrieved |

---

## 🏛️ Design Decisions

### Chunking (chunk_size=1000, overlap=150)
- 1000 chars ≈ 200-250 tokens — well within embedding model limits
- Captures complete scientific arguments without splitting equations
- 150-char overlap prevents context loss at chunk boundaries
- `RecursiveCharacterTextSplitter` respects paragraph → sentence → word hierarchy

### Retrieval (top_k=5)
- k=3 misses context spread across multiple paper sections
- k=5 provides enough context without overwhelming the LLM
- k=8+ increases hallucination risk from conflicting context

### Memory (ConversationBufferMemory)
- Stores full conversation for natural follow-up questions
- Production upgrade: use `ConversationSummaryMemory` to cap token growth

### Embeddings (text-embedding-3-small)
- 62.3% MTEB score vs ada-002's 61.0% at 5x lower cost
- 8191 token context window — ideal for research text
- HuggingFace fallback for multilingual/offline scenarios

---

## 🔮 Future Improvements

- [ ] **Hybrid search** — BM25 + dense embeddings for better acronym/keyword recall
- [ ] **Query rewriting** — HyDE (Hypothetical Document Embeddings) for better retrieval
- [ ] **Metadata filtering** — Filter by document, author, year, section
- [ ] **Figure/table extraction** — PyMuPDF for multimodal understanding
- [ ] **Multi-user sessions** — Redis-backed session management
- [ ] **Streaming responses** — WebSocket streaming for real-time generation
- [ ] **Docker deployment** — Containerized with docker-compose
- [ ] **Pinecone/Qdrant** — Production vector database upgrade
- [ ] **Re-ranking** — Cohere Rerank or cross-encoder for precision improvement
- [ ] **Cost tracking** — Token usage and cost per query logging

---

## 🧑‍💻 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-3-small |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| Frontend | Streamlit |
| Backend API | FastAPI |
| PDF Parsing | PyPDF |
| Evaluation | RAGAS |
| Language | Python 3.10+ |

---

## 📄 License

MIT License. Built for educational and portfolio purposes.
