import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import RateLimitError

# Load OpenAI API key from Streamlit Secrets
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("🔐 Missing OpenAI API key. Please set it in Streamlit Cloud secrets.")
    st.stop()

llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")

# Streamlit UI setup
st.set_page_config(page_title="Personality Analyzer", page_icon="🧠")
st.title("🧠 Kath's AI Personality Analyzer")
st.subheader("Note: Requires an OpenAI Paid Plan")
st.subheader("[API Pricing](https://openai.com/api/pricing/)")
st.markdown("Paste your writing sample below and discover your personality profile based on tone, traits, and helpful suggestions.")

user_text = st.text_area("✍️ Your text:", height=250)

if st.button("🔍 Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
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

            try:
                result = llm.invoke(messages)
                st.subheader("📋 Results")
                st.markdown(result.content)
            except RateLimitError:
                st.error("🚫 You’ve exceeded your OpenAI API quota. Please visit https://platform.openai.com/account/usage to check your usage or upgrade your plan.")


