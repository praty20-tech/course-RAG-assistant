# app/streamlit_app.py
import streamlit as st
import sys
import os

# Ensure we can import from src even if running from a different folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RAG

st.set_page_config(page_title="Course RAG Assistant", layout="wide")
st.title("Course RAG Assistant (SBERT + FAISS)")

# --- FIX: CACHING THE MODEL ---
@st.cache_resource
def load_rag_engine():
    """Load the heavy RAG engine only once."""
    return RAG()

try:
    with st.spinner("Loading RAG engine... (this happens only once)"):
        rag = load_rag_engine()
except Exception as e:
    st.error(f"Failed to load RAG engine. Make sure you ran 'ingest' and 'embed_index' first.\nError: {e}")
    st.stop()
# ------------------------------

with st.sidebar:
    st.write("## Settings")
    st.write("Instructions:")
    st.markdown("- Put PDFs in `data/pdfs`")
    st.markdown("- Run ingest + embed scripts first.")
    # Mention Groq since that's what we are using now
    st.markdown("- **Model:** Groq (Llama 3.1)") 

query = st.text_input("Ask a question about the uploaded course materials:")

# Added columns for better layout
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.number_input("Passages to retrieve", min_value=1, max_value=10, value=5)
with col2:
    search_clicked = st.button("Search", use_container_width=True)

if search_clicked:
    if not query:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            # The actual RAG call
            out = rag.answer(query, top_k=top_k)
        
        # Display Answer
        st.success("Analysis Complete")
        st.markdown("### 🤖 Answer")
        st.write(out["answer"])
        
        st.divider()
        
        # Display Sources
        st.markdown("### 📄 Retrieved Context")
        for i, p in enumerate(out["passages"]):
            with st.expander(f"Source {i+1}: {p.get('source', 'Unknown')} (Score: {p.get('score', 0):.2f})"):
                st.info(f"ID: {p['id']}")
                st.write(p["text"])