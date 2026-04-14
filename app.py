import streamlit as st
import fitz
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import os

# ---- CONFIG ----
st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("🤖 Welcome To Data Scientist Rohit Kumar Chatbot")
st.image("rohit.jpg", width=300)

# ---- API KEY (USE ENV VARIABLE) ----
genai.configure(api_key= "AIzaSyBNdHBogx7kMKGEcW2gZR_Pf-oCM4k9Iq8")

# ---- LOAD RESUME ----
@st.cache_data
def extract_resume_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

resume_text = extract_resume_text("Rohit Kumar Resume.pdf")

# ---- SPLIT RESUME INTO CHUNKS (IMPORTANT) ----
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

resume_chunks = chunk_text(resume_text)

# ---- INIT SESSION ----
if "chat" not in st.session_state:
    model = genai.GenerativeModel("gemini-2.0-flash")
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.messages = []

# ---- DISPLAY HISTORY ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- USER INPUT ----
user_input = st.chat_input("Ask something about Rohit's resume...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔍 Retrieve relevant chunks (simple RAG)
    relevant_text = ""
    for chunk in resume_chunks:
        if any(word.lower() in chunk.lower() for word in user_input.split()):
            relevant_text += chunk + "\n"

    # fallback if nothing found
    if not relevant_text:
        relevant_text = resume_chunks[0]

    prompt = f"""
    You are a professional resume assistant.

    Answer based only on the following resume content:
    {relevant_text}

    Question: {user_input}
    """

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat.send_message(prompt)
                answer = response.text
            except ResourceExhausted:
                answer = "⚠️ API quota exceeded. Please try again later."

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---- OPTIONAL: CLEAR CHAT BUTTON ----
if st.button("🔄 Reset Chat"):
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.messages = []
