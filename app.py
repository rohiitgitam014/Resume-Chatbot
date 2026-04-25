import streamlit as st
from groq import Groq
from pypdf import PdfReader

# ---- CONFIG ----
st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("🤖 Welcome To Data Scientist Rohit Kumar Chatbot")
st.image("rohit.jpg", width=300)

# ---- API KEY ----
api_key = "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"
client = Groq(api_key=api_key)

# ---- LOAD RESUME FROM LOCAL PDF ----
PDF_PATH = "Rohit Kumar Resume.pdf"

@st.cache_data
def load_resume() -> str:
    reader = PdfReader(PDF_PATH)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

resume = load_resume()

# ---- SYSTEM PROMPT ----
def build_system_prompt(resume: str) -> str:
    return f"""You are a professional and friendly assistant for a resume chatbot.
Answer questions ONLY based on the resume provided below.
Be complete, accurate, and well-structured. Never skip any detail.

Rules:
- Always list ALL items when asked (never skip any entry)
- Use bullet points for readability
- If not mentioned in resume, say: "This detail isn't mentioned in the resume."
- Never make up or assume any information
- Always refer to the candidate by their name as found in the resume

RESUME:
{resume}
"""

# ---- SESSION ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- DISPLAY CHAT HISTORY ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- INPUT ----
user_input = st.chat_input("Ask something about the resume...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=800,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": build_system_prompt(resume)},
                        {"role": "user", "content": user_input}
                    ]
                )
                answer = completion.choices[0].message.content
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
