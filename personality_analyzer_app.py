import os
import re
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from openai import APIError, AuthenticationError, RateLimitError, NotFoundError, BadRequestError

st.set_page_config(page_title="Personality Analyzer", page_icon="🧠")
st.title("🧠 Kath's AI Personality Analyzer")

st.write("Note: Requires an OpenAI Paid Plan "+"[API Pricing](https://openai.com/api/pricing/)")
st.divider()

# --- API key ---
api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
if not api_key:
    st.error("🔐 Missing OpenAI API key. Add OPENAI_API_KEY in Streamlit → Settings → Secrets.")
    st.stop()

# --- Model + temperature controls ---
st.sidebar.header("⚙️ Settings")
model = st.sidebar.selectbox("Model", ["gpt-5-nano", "gpt-4o-mini"], index=0)

# IMPORTANT: gpt-5-nano only supports temperature = 1 (default)
if model == "gpt-5-nano":
    temp = 1.0
    st.sidebar.caption("`gpt-5-nano` uses fixed temperature = 1.0")
else:
    temp = st.sidebar.slider("Temperature", 0.0, 1.2, 0.7, 0.1)

# Build the LLM instance (use Responses API route for newer models)
llm = ChatOpenAI(
    api_key=api_key,
    model=model,
    temperature=temp,
    use_responses_api=True,   # ensure Responses API path
)

def _to_markdown(result) -> str:
    """Normalize LangChain result.content into clean markdown text."""
    # Typical LangChain return is an AIMessage
    if isinstance(result, AIMessage):
        parts = result.content
        # If it's already a string, done
        if isinstance(parts, str):
            return parts.strip()
        # If it's a list of content parts, pull out the text segments
        if isinstance(parts, list):
            out = []
            for p in parts:
                # p might be a dict {'type':'text','text':'...'}
                if isinstance(p, dict) and p.get("type") == "text":
                    out.append(p.get("text", ""))
                # or a LangChain content object with .type/.text
                elif hasattr(p, "type") and getattr(p, "type", None) == "text":
                    out.append(getattr(p, "text", ""))
            return "\n\n".join([t for t in out if t]).strip()
    # Fallback: just string it
    return str(result).strip()


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
                "1) Emotional tone\n"
                "2) Big Five personality trait estimates (O,C,E,A,N)\n"
                "3) A likely MBTI type\n"
                "4) Personalized advice, plus 2 book or career recommendations\n"
                "...Be concise, accurate, and thoughtful. **Return Markdown only.**"
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_text),
            ]

            try:
                result = llm.invoke(messages)
                st.subheader("📋 Results")
                clean_md = _to_markdown(result)

                clean_md = re.sub(r"(?m)^\s*1\)\s*(.+)$", r"### 1) \1", clean_md)
                clean_md = re.sub(r"(?m)^\s*2\)\s*(.+)$", r"### 2) \1", clean_md)
                clean_md = re.sub(r"(?m)^\s*3\)\s*(.+)$", r"### 3) \1", clean_md)
                clean_md = re.sub(r"(?m)^\s*4\)\s*(.+)$", r"### 4) \1", clean_md)

                st.markdown(clean_md)
            except BadRequestError as e:
                # Show full server message so you can see which param was rejected
                st.error(f"Bad request: {e}")
            except NotFoundError:
                st.error("Model not found for this key. Try switching to gpt-4o-mini.")
            except AuthenticationError:
                st.error("🔑 Authentication failed. Double-check OPENAI_API_KEY.")
            except RateLimitError:
                st.error("🚫💸 You have exceeded your OpenAI API credits. Please visit https://platform.openai.com/account/usage to check your usage or upgrade your plan.")
            except APIError as e:
                st.error(f"OpenAI API error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

                
                
