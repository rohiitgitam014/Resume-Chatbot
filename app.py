import streamlit as st
import fitz  # PyMuPDF

import re

# ---- CONFIG ----
st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("🤖 Welcome To Data Scientist Rohit Kumar Chatbot")
st.image("rohit.jpg", width=300)

# ---- API KEY ----
from dotenv import load_dotenv
import os

load_dotenv()  # load .env file

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# ---- LOAD RESUME ----
@st.cache_data
def extract_resume_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

resume_text = extract_resume_text("Rohit Kumar Resume.pdf")

# ---- CLEAN TEXT ----
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove weird chars
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

cleaned_resume = clean_text(resume_text)

# ---- CHUNKING ----
def chunk_text(text, chunk_size=2000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = chunk_text(cleaned_resume)

# ---- SESSION INIT ----
if "chat" not in st.session_state:
    model = genai.GenerativeModel("gemini-2.5-flash-lite")  # stable model
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.messages = []

    # ---- SYSTEM INSTRUCTION ----
    st.session_state.chat.send_message(
        "You are a professional resume assistant.\n"
        "Answer ONLY based on the resume provided.\n"
        "If the answer is not available, say: 'Not mentioned in resume'."
    )

    # ---- SEND RESUME IN CHUNKS ----
    for i, chunk in enumerate(chunks):
        st.session_state.chat.send_message(f"Resume Part {i+1}:\n{chunk}")

# ---- DISPLAY CHAT HISTORY ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- USER INPUT ----
user_input = st.chat_input("Ask something about Rohit's resume...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat.send_message(user_input)
            st.markdown(response.text)

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.text
    })
