import os
import time
import fitz
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="StudyMate", layout="wide")

# Background Gradient & Styling
st.markdown("""
<style>
body {
  background: linear-gradient(to bottom right, #f0f4f8, #dbe9ff);
}
.answer-box {
  background: #e6ffe6;
  border: 1px solid #b3ffb3;
  border-radius: 12px;
  padding: 14px;
}
.chunk-box {
  background: #eef;
  border-radius: 8px;
  padding: 10px;
  margin-top: 8px;
  font-size: 0.9em;
}
.sidebar-title {
  color: #4B8BBE;
  font-weight: bold;
  font-size: 18px;
}
.footer {
  text-align: center;
  font-size: 0.8em;
  color: gray;
  margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# Logo & Title
st.image("https://copilot.microsoft.com/th/id/BCO.653c6183-47ca-4ed0-9d41-7dc716dd0871.png", width=180)
st.markdown("<h1 style='text-align:center;'>📘 StudyMate – Academic PDF Q&A Assistant</h1>", unsafe_allow_html=True)

# ------------------ Cache & Helpers ------------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500, overlap=80):
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text): break
        start = end - overlap
    return chunks

@st.cache_resource
def build_faiss_index(chunks, _embedder: SentenceTransformer):
    if not chunks: return None, None
    batch_size = 32
    embs = []
    progress = st.progress(0)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embs = _embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.extend(batch_embs)
        progress.progress(min((i + batch_size) / len(chunks), 1.0))
    embs = torch.tensor(embs).numpy()
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    progress.empty()
    return index, embs

@st.cache_resource
def load_granite(hf_token: str):
    model_id = "ibm-granite/granite-3.3-2b-instruct"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, device_map="auto", torch_dtype=dtype
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return tokenizer, model, generator

def get_hf_token():
    return st.session_state.get("hf_token") or os.getenv("HF_TOKEN", "")

# ------------------ Session State Initialization ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🔧 Settings / Login</div>", unsafe_allow_html=True)

    # Hugging Face Token
    hf_token_input = st.text_input("HF Token", type="password", placeholder="hf_...")
    if hf_token_input:
        st.session_state.hf_token = hf_token_input

    st.markdown("---")
    # Simple Login
    if not st.session_state.logged_in:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if username.strip() and password.strip():
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("Enter valid credentials!")
    else:
        st.success("Logged in ✅")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown("---")
    st.markdown("<div class='sidebar-title'>📘 About StudyMate</div>", unsafe_allow_html=True)
    st.markdown("""
- Purpose: Academic Q&A assistant for PDFs  
- Tech Stack: Streamlit, FAISS, Hugging Face, IBM Granite  
- Features: Semantic search, traceable answers, multi-chunk context, session history  
- Built at: Cognitive X Hackathon  
""")

# ------------------ Stop if not logged in ------------------
if not st.session_state.logged_in:
    st.stop()

# ------------------ File Upload ------------------
uploaded_files = st.file_uploader("Upload one or more academic PDFs", type="pdf", accept_multiple_files=True)

full_text = ""
if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for up in uploaded_files:
        pdf_path = os.path.join("data", up.name)
        with open(pdf_path, "wb") as f:
            f.write(up.getbuffer())
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if page_text:
                full_text += f"[Page {page_num}]\n{page_text}\n"

if uploaded_files and not full_text.strip():
    st.warning("⚠ No selectable text found. Use text-based PDFs for best results.")

# ------------------ Build Embeddings ------------------
chunks, index, embedder = [], None, None
if full_text.strip():
    embedder = get_embedder()
    chunks = chunk_text(full_text, chunk_size=500, overlap=80)
    with st.spinner("🔍 Building FAISS index... Please wait."):
        index, _ = build_faiss_index(chunks, embedder)
    st.success(f"✅ Indexed {len(chunks)} chunks successfully!")

# ------------------ Q&A Loop ------------------
def qa_loop():
    user_q = st.text_input("💬 Ask a Question", placeholder="e.g., What’s in page 3?", key=f"q_{len(st.session_state.history)}")
    if user_q:
        if not uploaded_files:
            st.error("Please upload at least one PDF first.")
        elif not full_text.strip() or index is None or not chunks:
            st.error("We couldn't build a search index. Try a different PDF.")
        else:
            hf_token = get_hf_token()
            if not hf_token:
                st.error("Please enter your Hugging Face token in the sidebar.")
            else:
                with st.spinner("🤔 Generating your answer... Please wait..."):
                    t0 = time.perf_counter()
                    q_emb = embedder.encode([user_q])
                    D, I = index.search(q_emb, k=3)
                    selected_chunks = [chunks[i] for i in I[0]]
                    combined_context = "\n\n".join(selected_chunks)

                    prompt = (
                        f"Answer the user's question using the following context. Be concise and accurate.\n\n"
                        f"Context:\n{combined_context}\n\n"
                        f"Question: {user_q}\nAnswer:"
                    )

                    tokenizer, model, generator = load_granite(hf_token)
                    out = generator(
                        prompt,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id
                    )[0]["generated_text"]

                    answer = out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
                    elapsed = time.perf_counter() - t0

                st.success("✅ Answer Ready")
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
                st.caption(f"Generated in {elapsed:.1f}s using IBM Granite 3.3 2B Instruct")

                with st.expander("📄 Show Source Chunks"):
                    for i, chunk in enumerate(selected_chunks, start=1):
                        st.markdown(f"<div class='chunk-box'><b>Chunk {i}:</b><br>{chunk}</div>", unsafe_allow_html=True)

                st.session_state.history.append((user_q, answer))

# ------------------ Display History ------------------
if st.session_state.history:
    with st.expander("🕘 View Previous Questions"):
        for idx, (q, a) in enumerate(reversed(st.session_state.history), start=1):
            st.markdown(f"Q{idx}: {q}")
            st.markdown(f"<div class='answer-box'>{a}</div>", unsafe_allow_html=True)

# ------------------ Run Q&A Loop ------------------
qa_loop()

# ------------------ Footer Badge ------------------
st.markdown("<div class='footer'>🚀 Built at Cognitive X Hackathon</div>", unsafe_allow_html=True)
