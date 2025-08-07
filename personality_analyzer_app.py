import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load OpenAI API key from Streamlit Secrets
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OpenAI API key.")
    st.stop()
    
llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
#llm = ChatOpenAI(api_key=api_key, model="gpt-4")

# Streamlit UI setup
st.set_page_config(page_title="Personality Analyzer", page_icon="🧠")
st.title("🧠 AI Personality Analyzer")
st.markdown("Paste your writing sample below and discover your personality profile based on tone, traits, and helpful suggestions.")

user_text = st.text_area("✍️ Your text:", height=250)

if st.button("🔍 Analyze"):
    with st.spinner("Thinking..."):
        system_prompt = (
            "You are a psychologist trained in the Big Five and MBTI. "
            "Analyze the user's writing and return:\n"
            "1. Emotional tone\n"
            "2. Big Five personality trait estimates\n"
            "3. A likely MBTI type\n"
            "4. Personalized advice, book or career recommendations\n"
            "Be concise, accurate, and thoughtful."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text)
        ]

        result = llm.invoke(messages)
        st.subheader("📋 Results")
        st.markdown(result.content)

