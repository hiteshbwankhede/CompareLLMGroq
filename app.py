import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import os

# Optional: Enable LangSmith tracing (if secrets are set)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Streamlit UI
st.set_page_config(page_title="LLM Model Comparator", layout="centered")
st.title("🔍 Compare Two LLMs")
st.markdown("Enter a prompt and compare outputs from **two different Groq models** side by side.")

# Input prompt
prompt = st.text_input("Enter a prompt:", value="Tell me a short story about the moon.")

# Temperature control
temperature = st.slider("Select temperature:", 0.0, 1.0, 0.7, 0.1)

# Model selection
model_options = [
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-guard-4-12b"
]

col1, col2 = st.columns(2)
with col1:
    model_1 = st.selectbox("Model 1", model_options, index=0)
with col2:
    model_2 = st.selectbox("Model 2", model_options, index=1)

# Prompt template
template = PromptTemplate.from_template("Tell me a story about:{user_prompt}")

# Run when button clicked
if st.button("🔍 Compare"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating responses..."):

            llm_1 = ChatGroq(model_name=model_1, temperature=temperature)
            llm_2 = ChatGroq(model_name=model_2, temperature=temperature)

            messages = [HumanMessage(content=template.format(user_prompt=prompt))]

            response_1 = llm_1.invoke(messages)
            response_2 = llm_2.invoke(messages)

        st.markdown("### 🔁 Comparison Results")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🧠 {model_1}")
            st.write(response_1.content.strip())
        with col2:
            st.subheader(f"🤖 {model_2}")
            st.write(response_2.content.strip())