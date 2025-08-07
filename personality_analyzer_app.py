import streamlit as st
import os
from crewai import Agent, Crew, Task
from langchain-openai.chat_models import ChatOpenAI

# Load API Key from environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set your OpenAI API key in Streamlit Cloud secrets.")
    st.stop()

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model='gpt-4')

# Define the agents
text_analyzer = Agent(
    role='Tone & Emotion Analyzer',
    goal='Identify sentiment, emotional tone, and writing style',
    backstory='An expert in psychology and linguistics.',
    llm=llm,
)

personality_profiler = Agent(
    role='Personality Profiler',
    goal='Infer Big Five and MBTI types based on writing',
    backstory='A seasoned psychological profiler.',
    llm=llm,
)

recommender = Agent(
    role='Lifestyle Recommender',
    goal='Give personalized self-help, career, and reading suggestions based on personality traits',
    backstory='A life coach and mentor.',
    llm=llm,
)

# Define tasks (we will inject user_text later)
tasks = [
    Task(agent=text_analyzer, description="Analyze tone and emotional content."),
    Task(agent=personality_profiler, description="Generate a personality profile."),
    Task(agent=recommender, description="Provide recommendations based on profile.")
]

crew = Crew(agents=[text_analyzer, personality_profiler, recommender], tasks=tasks)

# Streamlit UI
st.title("🧠 AI Personality Analyzer")

user_text = st.text_area("Paste a writing sample (e.g. journal entry, email):", height=250)

if st.button("Analyze") and user_text:
    with st.spinner("Analyzing with your AI crew..."):
        for task in crew.tasks:
            task.input = user_text
        results = crew.run()
        st.subheader("🧾 Results")
        for r in results:
            st.markdown(f"### {r['task']}")
            st.write(r['output'])
else:
    st.info("Paste some text and click Analyze.")

