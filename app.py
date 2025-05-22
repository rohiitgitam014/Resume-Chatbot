import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai

# ---- CONFIG ----
st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("🤖Welcome To  Data Scientists Rohit kumar Chatbot ")
st.image("rohit.jpg",width=240, caption="Rohit Kumar – Data Scientist") 

# ---- GEMINI API KEY INPUT ----
api_key = "AIzaSyCPQ_eQIm-Csx6yYhchWDzbxqZQ1N5-zd0"

genai.configure(api_key=api_key)

# ---- RESUME LOADING ----
@st.cache_data
def extract_resume_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

resume_text = extract_resume_text("Rohit Kumar.pdf")

# ---- SESSION STATE INITIALIZATION ----
if "chat" not in st.session_state:
    model = genai.GenerativeModel("gemini-2.0-flash")
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.chat.send_message(f"This is the resume of Rohit Kumar:\n\n{resume_text}")
    st.session_state.messages = []

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

    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": response.text})
