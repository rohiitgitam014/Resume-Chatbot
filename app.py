import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# ---- CONFIG ----
st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("🤖 Welcome To Data Scientist Rohit Kumar Chatbot")
st.image("rohit.jpg", width=300)

# ---- API KEY ----
api_key = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your actual Groq API key
client = Groq(api_key=api_key)

# ---- LOAD PDF ----
@st.cache_data
def extract_resume_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ---- CLEAN TEXT ----
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---- CHUNKING ----
def chunk_text(text, chunk_size=400):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---- BUILD FAISS INDEX ----
@st.cache_resource
def build_faiss_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, model

# ---- LOAD & PREPARE ----
resume_text = extract_resume_text("Rohit Kumar Resume.pdf")
cleaned_resume = clean_text(resume_text)
chunks = chunk_text(cleaned_resume)
faiss_index, embed_model = build_faiss_index(chunks)

# ---- SEMANTIC SEARCH ----
def get_relevant_context(user_input, top_k=5):
    query_embedding = embed_model.encode([user_input], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    top_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return " ".join(top_chunks)[:4000]

# ---- PROMPT ----
def build_prompt(user_input, context):
    return f"""You are a friendly and professional assistant representing Rohit Kumar's resume.

Answer questions about Rohit Kumar based ONLY on the resume content provided below.

Guidelines:
- Give clear, well-structured, and friendly answers
- Use bullet points or formatting where it improves readability
- If asked about education, mention degree, institution, year, and grades if available
- If asked about experience, mention company, role, duration, and key responsibilities
- If asked about skills, list them neatly by category if possible
- If the information is truly not in the resume, say: "This detail isn't mentioned in Rohit's resume."
- Never make up or assume information not present in the resume
- Always refer to the person as "Rohit" in your answers

Resume Content:
{context}

Question: {user_input}
Answer:"""

# ---- SESSION ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- DISPLAY CHAT HISTORY ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- INPUT ----
user_input = st.chat_input("Ask something about Rohit's resume...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                context = get_relevant_context(user_input)
                prompt = build_prompt(user_input, context)
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=700,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = completion.choices[0].message.content
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
