import streamlit as st
from groq import Groq
from pypdf import PdfReader
import re

# ---- CONFIG ----
st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("🤖 Welcome To Data Scientist Rohit Kumar Chatbot")
st.image("rohit.jpg", width=300)

# ---- API KEY ----
api_key = "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"  # Replace with your actual Groq API key
client = Groq(api_key=api_key)

# ---- FULL RESUME (hardcoded for 100% accuracy) ----
FULL_RESUME = """
ROHIT KUMAR
Location: Bengaluru, IN 560073
Phone: +91-8144185441
Email: rohitgitam014@gmail.com

PROFESSIONAL SUMMARY:
Detail-oriented Data Scientist with expertise in machine learning, predictive modelling, NLP, and data visualization. Passionate about leveraging AI-driven solutions to extract actionable insights and drive business growth. Proficient in Python, SQL, Power BI, and Tableau, with a strong foundation in real-time data processing and analytics.

SKILLS & TOOLS:
- Programming & Data Science: Python, SQL, Machine Learning, NLP, Deep Learning, Time Series Forecasting, Feature Engineering, Dimensionality Reduction (PCA), Data Preprocessing (StandardScaler, OneHotEncoder, LabelEncoder), Feature Selection, Model Evaluation & Selection (Cross-validation, GridSearchCV, Stratified K-Fold), Imbalanced Data Handling (SMOTE, class weights)
- Data Analytics & Visualization: Tableau, Power BI, EDA, Predictive Modeling, Statistical Modeling
- Business & Communication: Client Presentations, Business Understanding, Report Generation
- Frameworks & Tools: Scikit-learn, Seaborn, Pandas, NumPy, Hadoop
- Math & Stats: Linear Algebra, Probability & Statistics, Gradient Descent, Hypothesis Testing, Regularization, Bayes Theorem, Central Limit Theorem, Statistical Inference, Correlation & Covariance, Information Theory (Entropy)

PROFESSIONAL EXPERIENCE:

1. Data Scientist | Palette – Artful Craft, Colorful Tech (The AI-Powered Creative Solutionists)
   Duration: Aug 2025 – Feb 2026
   - Applied Python-based Data Science techniques to analyze the Hazzys Fashion retail dataset and build ML models for Total Sales prediction.
   - Conducted EDA to understand customer purchase behavior, product-level sales distribution, pricing impact, and key drivers influencing Total Sales.
   - Performed data preprocessing and feature engineering including missing value treatment, categorical encoding, and numerical feature scaling.
   - Built and optimized a Random Forest Regressor to predict Total Sales across multiple product categories.
   - Evaluated model using R², RMSE, MSE, MAE, and MAPE.
   - Translated insights into actionable business recommendations supporting inventory planning, demand estimation, and revenue optimization.

2. Data Scientist | Ahana Systems and Solutions
   Duration: Jan 2025 – July 2025
   - Developed predictive models focused on machine learning to address real-world business challenges across finance and operations.
   - Performed data preprocessing, feature engineering, feature selection, and model optimization using Pandas, Scikit-learn, and NumPy.
   - Built and deployed a stock price prediction model using Last Traded Price (LTP) as the target variable.
   - Applied Random Forest to predict LTP movement for investment risk decisions.
   - Validated model using R-squared, MAE, and MSE; visualized results using Seaborn and Matplotlib.

3. Data Science Intern | BDreamz Global Solutions Private Limited
   Duration: Jun 2023 – Aug 2024
   - Built a classification model using Python to solve a real-world business prediction problem.
   - Conducted EDA to uncover patterns and relationships in structured data.
   - Performed data preprocessing including missing value handling, encoding, and feature scaling using StandardScaler.
   - Addressed imbalanced class distribution using SMOTE.
   - Trained and evaluated models using Logistic Regression, achieving 87% recall score.
   - Assessed model using Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.

4. Data Science Capstone Project | UpGrad | Uber India Systems Private Limited (Remote)
   Duration: Nov 2023 – Feb 2024
   - Solved a real-world business problem using Python, ML, and end-to-end data science methodologies.
   - Followed the CRISP-DM framework from problem understanding to model evaluation.
   - Conducted EDA using Pandas, Seaborn, and Matplotlib.
   - Trained multiple regression models; achieved 89% accuracy (R² score) with Linear Regression.

PROJECTS:

1. Stock Market Price Prediction (GitHub)
   - Analyzed historical stock price data (Open, High, Low, Close, Volume) to identify market patterns.
   - Built a Random Forest regression model to predict Last Traded Price (LTP).
   - Performed EDA, feature engineering, outlier treatment, correlation analysis, and scaling.
   - Evaluated using R-squared, MAE, and MSE.
   - Delivered risk management insights by identifying price volatility patterns.

2. AI Chatbot using NLP (GitHub)
   - Developed an AI-powered chatbot using NLP techniques.
   - Implemented text preprocessing including tokenization, normalization, and lemmatization using NLTK.
   - Designed an intent recognition pipeline to classify user queries.
   - Combined rule-based logic with pre-trained language models.
   - Created structured dialogue flow for multi-turn conversations.

3. Fake URL Detection using LSTM
   - Built a phishing detection system using Deep Learning (LSTM).
   - Collected and processed real-world URL datasets from PhishTank.
   - Converted textual URL data into numerical sequences using tokenization and embedding.
   - Designed and trained an LSTM neural network to capture sequential URL patterns.
   - Evaluated using confusion matrix, precision, recall, F1-score, and ROC-AUC.

EDUCATION & CERTIFICATIONS:
1. Certified Data Scientist | University of Texas
2. Post Graduate in Data Science & Business Analytics | Great Learning
3. Graduate Certificate in Data Science | UpGrad
4. B.Tech in Computer Science & Engineering | GITAM Institute of Science and Technology
"""

# ---- SYSTEM PROMPT ----
SYSTEM_PROMPT = f"""You are a professional and friendly assistant for Rohit Kumar's resume chatbot.

Answer questions ONLY based on the resume provided below. 
Be complete, accurate, and well-structured. Never skip any detail.

Rules:
- Always list ALL items when asked (never skip any entry)
- Use bullet points for readability
- If not mentioned in resume, say: "This detail isn't mentioned in Rohit's resume."
- Never make up or assume any information
- Always refer to him as "Rohit"
- For education, always list all 4 entries exactly as written

ROHIT KUMAR'S COMPLETE RESUME:
{FULL_RESUME}
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
            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=800,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
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
