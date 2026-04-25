import streamlit as st
from groq import Groq
from pypdf import PdfReader
import re
import os

# ---- CONFIG ----
st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("🤖 Welcome To Data Scientist Rohit Kumar Chatbot")
st.image("rohit.jpg", width=300)

# ---- API KEY ----
api_key =  os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ API key not found.")
    st.stop()

client = Groq(api_key=api_key)

# ---- LOAD PDF ----
@st.cache_data
def extract_resume_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

resume_text = extract_resume_text("Rohit Kumar Resume.pdf")

# ---- CLEAN TEXT ----
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

cleaned_resume = clean_text(resume_text)

# ---- CHUNKING ----
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = chunk_text(cleaned_resume)

# ---- RELEVANT CHUNK ----
def get_relevant_chunk(user_input, chunks):
    scores = []
    for chunk in chunks:
        score = sum(word.lower() in chunk.lower() for word in user_input.split())
        scores.append(score)
    return chunks[scores.index(max(scores))]

# ---- PROMPT ----
def build_prompt(user_input, context):
    return f"""
You are a professional resume assistant.
Answer ONLY based on the resume below.
If answer is not available, say: 'Not mentioned in resume'.

Resume:
{context}

Question:
{user_input}
"""

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
            relevant_chunk = get_relevant_chunk(user_input, chunks)
            prompt = build_prompt(user_input, relevant_chunk)
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = completion.choices[0].message.content
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
