import streamlit as st
import os
from langchain_openai import ChatOpenAI

# Load OpenAI API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OpenAI API key.")
    st.stop()

llm = ChatOpenAI(api_key=api_key, model="gpt-4")

st.set_page_config(page_title="Personality Analyzer", page_icon="🧠")
st.title("🧠 AI Personality Analyzer")
st.markdown("Paste your writing sample below, and we'll analyze your emotional tone and personality traits.")

user_text = st.text_area("✍️ Your text:", height=250)

if st.button("🔍 Analyze"):
    with st.spinner("Thinking..."):
        system_prompt = (
            "You are an expert psychologist trained in the Big Five and MBTI. "
            "Analyze the following text and return:\n"
            "1. The emotional tone\n"
            "2. Big Five personality trait estimates\n"
            "3. A probable MBTI type\n"
            "4. Personalized self-help or book recommendations\n"
            "Be concise, specific, and thoughtful."
        )
        result = llm.invoke(user_text, system_message=system_prompt)
        st.subheader("📋 Results")
        st.markdown(result.content)
