import streamlit as st
import requests
import re


st.markdown(
    """
    <style>
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    @keyframes pulseGlow {
        0% {
            text-shadow: 0 0 5px rgba(0, 150, 255, 0.3);
        }
        50% {
            text-shadow: 0 0 20px rgba(0, 150, 255, 0.8);
        }
        100% {
            text-shadow: 0 0 5px rgba(0, 150, 255, 0.3);
        }
    }

    .stApp {
        background: linear-gradient(-45deg, #09091a, #151596, #911363, #2a043b);
        background-size: 400% 400%;
        animation: gradientShift 25s ease infinite;
        color: white;
    }

    .glow-text {
        display: inline-block;
        animation: pulseGlow 1s ease-in-out infinite;
        font-size: 10em;
        font-weight: bold;
        text-align: center;
        width: 100%;
    }
    </style>

    <h1 class="glow-text">SUMMA.RY</h1>
""", 
unsafe_allow_html=True
)




summaryPowerDict={
    0:"\n\nThis doesnt have anywhere near the level of detail it should, how horribly reductionist. Fixed it for you: ",
    1:"\n\nBasically: ",
    2:"\n\nTL;DR: ",
    3:"\n\n\n\nELI5: ",
    4:"\n\n\n\nOr, in 10 words: "
}

level = st.slider("Summary Power", 0, 4, 1, format="%d")
suffix = summaryPowerDict[level]


# ---- Sends just one message (latest) to local Qwen ----
def query_lm_studio(user_input, prompt_suffix):
    payload = {
        "model": "Qwen2.5-VL-3B-Instruct",
        "messages": [
            {"role": "user", "content": user_input + prompt_suffix}
        ]
    }

    try:
        response = requests.post("http://localhost:1234/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ Error contacting LLM:", e)
        print("⚠️ Response content (if any):", getattr(e, 'response', None))
        return "Error contacting language model."
    
    
# ---- Main App ----
#st.title("Summarization Tool")

if 'display_history' not in st.session_state:
    st.session_state.display_history = []

user_input = st.chat_input("Paste something to summarize...")
if user_input:
    st.session_state.display_history = []
    response = query_lm_studio(user_input, suffix)
    st.session_state.display_history.append({"role": "user", "content": user_input})
    st.session_state.display_history.append({"role": "assistant", "content": response})    

# ---- Display Summary ----
def clean_tldr(content):
    return re.sub(r"\s*TLDR:\s*", "", content, flags=re.IGNORECASE)

for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(clean_tldr(msg["content"]))