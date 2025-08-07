import streamlit as st
import os
from crewai import Agent, Crew, Task
from langchain import ChatOpenAI
#from langchain_openai import ChatOpenAI

# Load API Key from environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🔐 Please set your OpenAI API key in Streamlit Cloud secrets.")
    st.stop()

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model='gpt-4')

# Define the agents
text_analyzer = Agent(
    role='Tone & Emotion Analyzer',
    goal='Identify sentiment, emotional tone, and writing style',
    backstory='An expert in psychology and linguistics trained to assess mood and tone in language.',
    llm=llm,
)

personality_profiler = Agent(
    role='Personality Profiler',
    goal='Infer Big Five and MBTI types based on writing',
    backstory='A seasoned psychological profiler with expertise in behavioral analysis and psychometrics.',
    llm=llm,
)

recommender = Agent(
    role='Lifestyle Recommender',
    goal='Give personalized self-help, career, and reading suggestions based on personality traits',
    backstory='A life coach and career mentor who specializes in matching personality types to ideal resources.',
    llm=llm,
)

# Define the tasks
user_text = ""  # This will be updated dynamically in Streamlit

tasks = [
    Task(agent=text_analyzer, description="Analyze the tone, sentiment, and emotional content of the input text."),
    Task(agent=personality_profiler, description="Generate a Big Five and MBTI profile based on the text."),
    Task(agent=recommender, description="Suggest tailored advice, books, and career paths based on the personality profile."),
]

crew = Crew(
    agents=[text_analyzer, personality_profiler, recommender],
    tasks=tasks,
    verbose=False
)

# Streamlit UI
st.set_page_config(page_title="AI Personality Analyzer", page_icon="🧠", layout="centered")
st.markdown("""
    <style>
        .main { background-color: #fdf6f0; }
        .stButton > button { background-color: #ff8c42; color: white; font-weight: bold; }
        .stTextArea textarea { background-color: #fffaf0; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center;'>🧠 AI Personality Analyzer</h1>
    <p style='text-align: center; font-size: 1.1rem;'>
        Paste a journal entry, email, or any text sample, and let our AI team analyze your emotional tone, 
        infer your personality traits, and offer tailored recommendations.
    </p>
""", unsafe_allow_html=True)

user_text = st.text_area("✍️ Paste your writing sample:", height=250)

if st.button("🔍 Analyze My Personality") and user_text:
    with st.spinner("Analyzing with your AI crew..."):
        for task in crew.tasks:
            task.input = user_text
        results = crew.run()

        st.markdown("---")
        st.subheader("📋 Analysis Report")
        for r in results:
            st.markdown(f"### 🔹 {r['task']}")
            st.write(r['output'])
        st.markdown("---")
else:
    st.info("Enter a text sample above and click the button to begin.")
