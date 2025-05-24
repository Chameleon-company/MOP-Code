# Import Libraries
import streamlit as st
import google.generativeai as genai
import json

# Setup Gemini Flash
api_key = "'YOUR_API_KEY'"
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

#Optional Dataset Loading
try:
    with open("cleaned_data.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
except:
    dataset = []

#Response Generation Settings
generation_configs = {
    "Standard (default)": {"max_output_tokens": 200},
    "Creative (temp=0.9)": {"max_output_tokens": 200, "temperature": 0.9},
    "Precise (temp=0.3)": {"max_output_tokens": 200, "temperature": 0.3}
}

#Prompt Templates
prompt_styles = {
    "💙 Supportive Friend": lambda q: f"You are a caring and supportive friend. Your friend says: '{q}'\nHow would you kindly respond?",
    "🧠 Therapist": lambda q: f"You are a mental health therapist. Your patient shares: '{q}'\nRespond with empathy and guidance.",
    "🤖 Smart Assistant": lambda q: f"You are an intelligent AI assistant. The user says: '{q}'\nProvide a helpful response:"
}

#Initialize Conversation History
if 'history' not in st.session_state:
    st.session_state.history = []

#  Streamlit Interface
st.set_page_config(page_title="Mental Support App", page_icon="🧠")
st.title("🧠 Mental Health Assistant (Gemini Flash)")
st.write("Describe how you're feeling or what's troubling you:")

user_input = st.text_area("Your message", height=150)
prompt_style = st.radio("Choose response tone:", list(prompt_styles.keys()))
generation_setting = st.selectbox("Choose Response Style (Generation Settings)", list(generation_configs.keys()))
# Generate Response
if st.button("Get Response"):
    if user_input.strip() == "":
        st.warning("Please enter your message.")
    else:
        with st.spinner("Thinking..."):
            prompt = prompt_styles[prompt_style](user_input)
            config = generation_configs[generation_setting]
            try:
                response = model.generate_content(prompt, generation_config=config)
                reply = response.text.strip()
                st.success("🗨️ Response:")
                st.write(reply)

                st.session_state.history.append({
                    "input": user_input,
                    "style": prompt_style,
                    "response": reply
                })
            except Exception as e:
                st.error(f"An error occurred: {e}")
#Show Last Response
if st.session_state.history:
    last_reply = st.session_state.history[-1]["response"]
    with st.expander("📤 Share This Response"):
        st.code(last_reply, language="markdown")

 #Show Full Chat History
if st.session_state.history:
    all_history = "\n\n".join(
        [f"🗣️ {item['input']}\n🤖 ({item['style']}) {item['response']}" for item in st.session_state.history]
    )
    with st.expander("📋 Copy Full Conversation"):
        st.code(all_history, language="markdown")

 #Reset Chat
if st.button("🔄 Reset Conversation"):
    st.session_state.history = []
    st.success("Conversation has been reset.")
