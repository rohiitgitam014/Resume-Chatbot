import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
import re

# ---- CONFIG ----
st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("🤖 Welcome To Data Scientist Rohit Kumar Chatbot")
st.image("rohit.jpg", width=300)

# ---- GROQ CLIENT ----
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---- LOAD RESUME ----
@st.cache_data
def extract_resume_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

resume_text = extract_resume_text("Rohit Kumar Resume.pdf")

# ---- CLEAN TEXT ----
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

cleaned_resume = clean_text(resume_text)

# ---- PROMPT BUILDER ----
def build_prompt(user_input, resume_text):
    return f"""
You are a professional resume assistant.

Answer ONLY based on the resume below.
If answer is not available, say: 'Not mentioned in resume'.

Resume:
{resume_text}

Question:
{user_input}
"""

# ---- SESSION INIT ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- DISPLAY CHAT HISTORY ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- USER INPUT ----
user_input = st.chat_input("Ask something about Rohit's resume...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            prompt = build_prompt(user_input, cleaned_resume)

            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            answer = completion.choices[0].message.content
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
