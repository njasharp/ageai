import os
import streamlit as st
from typing import Dict, Optional, List
from groq import Groq

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Agents & AI ML methods", initial_sidebar_state="expanded")

# Supported models
SUPPORTED_MODELS: Dict[str, str] = {
    "Llama 3.2 1B (Preview)": "llama-3.2-1b-preview",
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it",
    "LLaVA 1.5 7B": "llava-v1.5-7b-4096-preview",
    "Llama 3.2 3B (Preview)": "llama-3.2-3b-preview",
    "Llama 3.2 11B Vision (Preview)": "llama-3.2-11b-vision-preview"
}

MAX_TOKENS: int = 1000

# Initialize Groq client with API key
@st.cache_resource
def get_groq_client() -> Optional[Groq]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please set it and restart the app.")
        return None
    return Groq(api_key=groq_api_key)

client = get_groq_client()

# Sidebar - Model Configuration
st.sidebar.image("p1a.png", width=280)
st.sidebar.text("highly advanced AI designed to ")
st.sidebar.text("simulate human-like Cognitive Tasks")
st.sidebar.title("Model Configuration")
selected_model = st.sidebar.selectbox("Choose an AI Model", list(SUPPORTED_MODELS.keys()))

# Sidebar - Temperature Slider
st.sidebar.subheader("Temperature")
temperature = st.sidebar.slider("Set temperature for response variability:", min_value=0.0, max_value=1.0, value=0.7)

# Sidebar - User Prompt Input
st.sidebar.subheader("User Prompt")
user_prompt = st.sidebar.text_area("Enter your user prompt:")

# Initialize session state for selected agent and agent output
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = None
if 'agent_output' not in st.session_state:
    st.session_state.agent_output = None

# Agent selection system prompt
agent_selection_prompt = """You are an expert agent investigator. Analyze the user prompt and determine the most suitable agent based on the following categories: Intent, Motor, Converse, Memory, Summarize, Goal, or Environment. Respond with only the single most appropriate category name."""

# Agent execution system prompts
agent_execution_prompts = {
    "Intent": "You are an Intent Generator Agent. Based on the user's prompt, generate a clear and concise intent statement.",
    "Motor": "You are a Motor Plan Agent. Based on the user's prompt, generate a high-level motor plan.",
    "Converse": "You are a Converse Agent. Based on the user's prompt, start a conversation that addresses their needs.",
    "Memory": "You are a Memory Manager Agent. Based on the user's prompt, access relevant memory and provide user context.",
    "Summarize": "You are a Summarization Agent. Based on the user's prompt, provide a concise summary of the key points.",
    "Goal": "You are a Goal Management Agent. Based on the user's prompt, update and manage relevant goals.",
    "Environment": "You are an Environment Feedback Agent. Based on the user's prompt, analyze and describe the user's environment.",
    "Default": "You are a versatile AI assistant. Analyze the user's prompt and provide an appropriate response based on your general knowledge and capabilities."
}

# Function to get response from Groq API
def get_groq_response(prompt: str, system_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=SUPPORTED_MODELS[selected_model],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

# Function to select the most appropriate agent
def select_agent(agent_response: str) -> str:
    valid_agents = list(agent_execution_prompts.keys())
    agents = [agent.strip() for agent in agent_response.split() if agent.strip() in valid_agents]
    return agents[0] if agents else "Default"

# Submit button for evaluation
if st.sidebar.button("Submit"):
    if client:
        # Step 1: Select the appropriate agent
        agent_response = get_groq_response(user_prompt, agent_selection_prompt)
        agent_type = select_agent(agent_response)
        st.session_state.selected_agent = agent_type
        
        # Step 2: Execute the user prompt with the selected agent
        execution_system_prompt = agent_execution_prompts[agent_type]
        st.session_state.agent_output = get_groq_response(user_prompt, execution_system_prompt)
    else:
        st.error("Groq client not initialized.")

# Main content area

# Display images
col1, col2 = st.columns(2)
with col1:
    st.image("p1.png")
    st.text ("....          mimic human cognitive methods") 
    st.markdown(
        """
        <h2>Agent Roles</h2>
        "Intent": <b>You are an Intent Generator Agent</b><br>
        "Motor": <b>You are a Motor Plan Agent</b><br>
        "Converse": <b>You are a Converse Agent</b><br>
        "Memory": <b>You are a Memory Manager Agent</b><br>
        "Summarize": <b>You are a Summarization Agent</b><br>
        "Goal": <b>You are a Goal Management Agent</b><br>
        "Environment": <b>You are an Environment Feedback Agent</b><br>
        "Default": <b>You are a versatile AI assistant.</b>
        """, unsafe_allow_html=True
    )
with col2:
    st.image("p3.jpg", width=500)

# Sidebar - Selected Agent Indicator (moved to main page sidebar)
st.sidebar.subheader("Selected Agent")
if st.session_state.selected_agent:
    st.sidebar.success(f"**{st.session_state.selected_agent} Agent**")
else:
    st.sidebar.info("No agent selected yet")

# Display agent output in the main content area
if st.session_state.agent_output:
    st.subheader("Agent Output")
    st.write(st.session_state.agent_output)

st.info("built by dw - create an autonomous, multi-functional AI assistant capable of thinking, planning, remembering, conversing, and summarizing.")
