"""
ui/streamlit_app.py
====================
Streamlit chat interface for the Research Paper Q&A Chatbot.

Layout:
  Left Sidebar: PDF uploader + document list + settings
  Main Area:    Chat history → user input → AI response → citations

State Management:
  - st.session_state.messages: full chat history (displayed in UI)
  - st.session_state.chain: the live RAG chain (with memory)
  - st.session_state.uploaded_files: names of processed documents
  - st.session_state.is_processing: lock to prevent double-submit
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from rag.loader import load_multiple_pdfs
from rag.chunker import chunk_and_clean
from rag.embeddings import get_embeddings
from rag.vector_store import build_vector_store, reset_collection, get_collection_stats
from rag.retriever import get_retriever
from rag.rag_chain import build_rag_chain, ask

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS Styling
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Citation boxes */
    .citation-box {
        background: #f0f4ff;
        border-left: 4px solid #4a6cf7;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .citation-title {
        font-weight: bold;
        color: #4a6cf7;
    }
    .citation-snippet {
        color: #555;
        font-style: italic;
        margin-top: 0.2rem;
    }
    /* Upload success */
    .success-badge {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.3rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem 0;
    }
    /* Processing spinner */
    .stSpinner > div {
        border-color: #4a6cf7 !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def process_pdfs(uploaded_files):
    """Save uploads to temp dir, process through full RAG pipeline, build chain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_paths = []
        for uf in uploaded_files:
            dest = Path(tmpdir) / uf.name
            dest.write_bytes(uf.getbuffer())
            temp_paths.append(dest)

        with st.spinner("📖 Loading PDFs..."):
            docs = load_multiple_pdfs(temp_paths)
            if not docs:
                st.error("Could not load any pages from the uploaded PDFs.")
                return

        with st.spinner("✂️ Chunking documents..."):
            chunks = chunk_and_clean(docs)
            st.session_state.all_chunks = chunks

        with st.spinner("🧠 Generating embeddings and indexing..."):
            try:
                embeddings = get_embeddings()
            except EnvironmentError as e:
                st.error(f"❌ Embedding error: {e}")
                return

            # Reset collection to avoid stale data on re-upload
            reset_collection(embeddings)
            vector_store = build_vector_store(chunks, embeddings)
            st.session_state.vector_store = vector_store

        with st.spinner("🔗 Building RAG chain..."):
            retriever = get_retriever(vector_store)
            chain = build_rag_chain(retriever)
            st.session_state.chain = chain

        st.session_state.uploaded_files = [uf.name for uf in uploaded_files]
        # Add welcome message
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"✅ I've indexed **{len(st.session_state.uploaded_files)} paper(s)** "
                    f"into **{len(chunks)} chunks**.\n\n"
                    "Ask me anything about the uploaded research papers! I'll answer with citations. 📄"
                ),
                "sources": [],
            }
        ]


def render_citation(source: dict, index: int):
    """Render a single citation box."""
    st.markdown(
        f"""<div class="citation-box">
            <div class="citation-title">
                [{index}] 📄 {source['file_name']} — Page {source['page']}
            </div>
            <div class="citation-snippet">"{source['snippet']}"</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Research Q&A")
    st.markdown("*Powered by RAG + GPT-4o-mini*")
    st.divider()

    # PDF Upload Section
    st.subheader("📁 Upload Papers")
    uploaded = st.file_uploader(
        "Upload academic PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research papers (arXiv-style PDFs work best)",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Process", type="primary", use_container_width=True):
            if not uploaded:
                st.warning("Please upload at least one PDF.")
            else:
                process_pdfs(uploaded)
                st.rerun()

    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chain:
                st.session_state.chain.memory.clear()
            st.rerun()

    st.divider()

    # Indexed Documents
    if st.session_state.uploaded_files:
        st.subheader("✅ Indexed Papers")
        for fname in st.session_state.uploaded_files:
            st.markdown(
                f'<div class="success-badge">📄 {fname}</div>',
                unsafe_allow_html=True,
            )

        if st.session_state.vector_store:
            stats = get_collection_stats(st.session_state.vector_store)
            st.caption(f"📊 {stats['total_chunks']} chunks indexed")

    st.divider()

    # Settings
    with st.expander("⚙️ Settings"):
        st.caption(f"**Model:** {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")
        st.caption(f"**Embeddings:** {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}")
        st.caption(f"**Chunk size:** {os.getenv('CHUNK_SIZE', '1000')}")
        st.caption(f"**Top-K retrieval:** {os.getenv('RETRIEVAL_TOP_K', '5')}")

    st.divider()
    st.caption("💡 **Tips:**\n- Ask follow-up questions naturally\n- Ask for specific page citations\n- Try 'Summarize the key findings'")


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────

st.title("🔬 Research Paper Q&A Chatbot")

if not st.session_state.chain:
    # Landing state — no documents indexed yet
    st.info("👈 Upload and process research papers in the sidebar to start chatting.")
    
    with st.container():
        st.markdown("### 💡 Example Questions")
        cols = st.columns(2)
        examples = [
            "What is the main contribution of this paper?",
            "What datasets were used for evaluation?",
            "How does their method compare to baselines?",
            "What are the limitations of this approach?",
            "Explain the methodology in simple terms.",
            "What are the key experimental results?",
        ]
        for i, ex in enumerate(examples):
            cols[i % 2].markdown(f"- *{ex}*")
else:
    # Chat interface
    chat_container = st.container()

    # Render message history
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

                # Show citations for assistant messages
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander(f"📚 View {len(msg['sources'])} Source(s)", expanded=False):
                        for i, source in enumerate(msg["sources"], 1):
                            render_citation(source, i)

    # Chat input
    if prompt := st.chat_input("Ask a question about the papers..."):
        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "sources": [],
        })

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching papers and generating answer..."):
                result = ask(st.session_state.chain, prompt)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 View {len(sources)} Source(s)", expanded=True):
                    for i, source in enumerate(sources, 1):
                        render_citation(source, i)

        # Store in history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
