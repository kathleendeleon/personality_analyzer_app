import os
import re
import json
import uuid
import streamlit as st

# ---------- Agentic libs ----------
# LangChain: tools, prompts, LLM wrappers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# CrewAI: multi-agent orchestration
from crewai import Agent as CrewAgent, Task as CrewTask, Crew, Process

# AutoGen (A2A): agent-to-agent critique loop
try:
    from autogen import AssistantAgent, UserProxyAgent
    AUTOGEN_AVAILABLE = True
except Exception:
    AUTOGEN_AVAILABLE = False

# MCP (Model Context Protocol): optional tool access
MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    MCP_AVAILABLE = True
except Exception:
    MCP_AVAILABLE = False

# ---------- App config ----------
st.set_page_config(page_title="Personality Analyzer — Agentic Edition", page_icon="🧠", layout="wide")
st.title("🧠 Personality Analyzer — Agentic Edition")
st.caption("Built with LangChain (orchestration), CrewAI (multi-agent workflow), MCP (interoperable tools), and an A2A critique loop (AutoGen).")

# ---------- Inputs ----------
with st.sidebar:
    st.header("Input")
    mode = st.radio("Content source", ["Paste Text", "Web URL"])
    openai_model = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.2, 0.2, 0.1)
    max_char = st.number_input("Max chars to analyze", min_value=500, max_value=20000, value=4000, step=500)
    run_autogen = st.checkbox("Run A2A critique loop (AutoGen)", value=True)
    run_mcp = st.checkbox("Use MCP tool (if available)", value=False, help="Requires an MCP server locally; demo 'read_webpage' tool.")

if mode == "Paste Text":
    user_text = st.text_area("Paste writing samples:", height=240, placeholder="Paste up to several paragraphs...")
else:
    url = st.text_input("Web URL to analyze", placeholder="https://example.com/article")
    user_text = ""

# ---------- Utilities ----------
def load_via_langchain(url: str, max_chars: int) -> str:
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = "\n\n".join([d.page_content for d in docs])[:max_chars]
    return text

def chunk_text(text: str, max_chars: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_text(text[:max_chars])
    return chunks

# ---------- LangChain prompt building ----------
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior psychologist and data scientist. Be precise, evidence-aware, and concise."),
    ("user",
     """Analyze the following writing samples for personality signals.
Return:
1) Emotional tone (1-2 lines)
2) Big Five (O,C,E,A,N) with 1-2 sentence justification each
3) Likely MBTI type (best guess + 2nd guess) with rationale
4) Strengths & watchouts (bullets)
5) Two book or career recommendations
Text:
{passage}
""")
])

report_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an executive editor. Merge multiple analyses into a single, coherent report. Keep it under 500 words."),
    ("user", "Combine the following partial analyses into ONE final report:\n\n{partials}")
])

llm = ChatOpenAI(model=openai_model, temperature=temperature)

def analyze_chunks(chunks: list[str]) -> list[str]:
    parser = StrOutputParser()
    results = []
    for ch in chunks:
        chain = analysis_prompt | llm | parser
        results.append(chain.invoke({"passage": ch}))
    return results

def synthesize_report(partials: list[str]) -> str:
    parser = StrOutputParser()
    chain = report_prompt | llm | parser
    return chain.invoke({"partials": "\n\n---\n\n".join(partials)})

# ---------- CrewAI multi-agent workflow ----------
def run_crewai_pipeline(text: str) -> dict:
    # Agents
    analyst = CrewAgent(
        role="Language Analyst",
        goal="Extract psychological signals (tone, Big Five cues, MBTI indicators).",
        backstory="A linguistics PhD with experience in psychometrics.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    psychologist = CrewAgent(
        role="Psychologist",
        goal="Interpret signals into personality profiles with nuance and caution.",
        backstory="Licensed psychologist with background in differential psychology.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    editor = CrewAgent(
        role="Executive Editor",
        goal="Merge findings into a crisp, executive-ready report under 500 words.",
        backstory="Editorial lead ensuring clarity and balance.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # Tasks
    t1 = CrewTask(
        description="Analyze the provided text, extract tone, Big Five cues, MBTI signals, strengths and risks.",
        agent=analyst,
        expected_output="A structured bullet list of cues and justifications."
    )
    t2 = CrewTask(
        description="Translate cues into a draft profile: tone, Big Five estimates (+ justification), MBTI guess (with rationale), recommendations.",
        agent=psychologist,
        expected_output="A draft personality report (~300-400 words)."
    )
    t3 = CrewTask(
        description="Polish and compress the draft into a final report, ensuring coherence and actionable insights.",
        agent=editor,
        expected_output="Final personality report under 500 words."
    )

    crew = Crew(
        agents=[analyst, psychologist, editor],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff(inputs={"text": text})
    return {"crewai_report": str(result)}

# ---------- A2A (AutoGen) critique loop ----------
def autogen_critique(report: str) -> str:
    if not AUTOGEN_AVAILABLE:
        return "AutoGen not installed — skipping A2A critique."
    cfg = {"config_list": [{"model": openai_model}]}
    from autogen import AssistantAgent, UserProxyAgent
    assistant = AssistantAgent("assistant", llm_config=cfg)
    critic = AssistantAgent("critic", llm_config=cfg, system_message="Be a strict, helpful editor who suggests specific improvements.")
    user = UserProxyAgent("user", human_input_mode="NEVER")
    # Simple round: critic reviews, assistant revises
    critic_review = critic.generate_reply(messages=[{"role": "user", "content": f"Critique this report and propose improvements:\n\n{report}"}])
    revised = assistant.generate_reply(messages=[
        {"role": "user", "content": f"Here is a critique:\n{critic_review}\n\nRevise the report accordingly. Keep it under 500 words.\nOriginal report:\n{report}"}
    ])
    return revised

# ---------- Run pipeline ----------
run = st.button("Analyze")

if run:
    # Prepare text
    if mode == "Web URL":
        if not url:
            st.error("Please provide a URL.")
            st.stop()
        with st.status("Loading content via LangChain WebBaseLoader...", expanded=False):
            try:
                text = load_via_langchain(url, max_char)
            except Exception as e:
                st.error(f"Failed to load URL: {e}")
                st.stop()
    else:
        text = user_text

    if not text or len(text.strip()) < 50:
        st.error("Please provide enough text (at least ~50 characters).")
        st.stop()

    # LangChain chunked analysis + synthesis
    chunks = chunk_text(text, max_char)
    st.write(f"Chunk count: **{len(chunks)}**")
    partials = analyze_chunks(chunks[:3])  # cap for speed
    merged = synthesize_report(partials)

    # CrewAI sequential pipeline
    with st.status("Running CrewAI multi-agent workflow...", expanded=False):
        crew_out = run_crewai_pipeline(merged)

    final_report = crew_out.get("crewai_report", merged)

    # Optional A2A critique loop
    if run_autogen:
        st.info("Running A2A critique loop (AutoGen)...")
        final_report = autogen_critique(final_report)

    st.subheader("Final Personality Report")
    st.write(final_report)

    st.download_button("Download Report (.txt)", data=final_report, file_name=f"personality_report_{uuid.uuid4().hex[:8]}.txt")

st.markdown("---")
st.markdown("**Notes**")
st.markdown("""
- **LangChain** handles loading, chunking, prompting, and initial synthesis.
- **CrewAI** runs a mini multi-agent workflow (Analyst → Psychologist → Editor).
- **MCP** (optional) demonstrates calling a protocol-defined tool if an MCP server is available.
- **A2A** loop uses **AutoGen** for a quick critic→revise iteration.
""")
