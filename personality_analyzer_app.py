# personality_analyzer_app.py
pip install crewai

import streamlit as st
from crewai import Agent, Crew, Task
from langchain.chat_models import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(model='gpt-4')

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
    verbose=True
)

# Streamlit UI
st.title("🧠 AI Personality Analyzer")

user_text = st.text_area("Paste your writing sample (journal entry, essay, etc.):", height=250)

if st.button("Analyze") and user_text:
    with st.spinner("Analyzing with AI agents..."):
        # Set input dynamically for each task
        for task in crew.tasks:
            task.input = user_text

        results = crew.run()

        st.subheader("🧾 Analysis Report")
        for output in results:
            st.markdown(f"### {output['task']}")
            st.write(output['output'])
else:
    st.info("Please paste a text sample and click 'Analyze' to get started.")
