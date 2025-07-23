## Import stuff


import streamlit as st
import requests
import os
from openai import OpenAI
import google.generativeai as genai


### Styling

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

    <h1 class="glow-text">RLHF Demo</h1>
""", 
unsafe_allow_html=True
)


# Load API keys from secrets or env
client = OpenAI(api_key=st.secrets["providers"]["openai_key"])
genai.configure(api_key=st.secrets["providers"]["gemini_key"])


def query_chatgpt(history):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in history
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def query_gemini(history):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history
    )
    response = model.generate_content(prompt)
    return response.text


# Send to N8N webhook
def send_to_rlhf(prompt, chatgpt, gemini, preferred):
    payload = {
        "prompt": prompt,
        "chatgpt": chatgpt,
        "gemini": gemini,
        "preferred": preferred
    }
    response = requests.post("http://localhost:5678/webhook-test/7c75d2f2-9be5-43ef-929b-5a3956124692", json=payload)
    st.toast(f"Webhook sent! Status: {response.status_code}")


prompt = st.chat_input("Type your prompt...")

if prompt:
    history = [{"role": "user", "content": prompt}]
    chatgpt_resp = query_chatgpt(history)
    gemini_resp = query_gemini(history)


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ChatGPT**")
        st.write(chatgpt_resp)
        if st.button("👍", key="gpt-up"):
            send_to_rlhf(prompt, chatgpt_resp, gemini_resp, preferred="chatgpt")

    with col2:
        st.markdown("**Gemini**")
        st.write(gemini_resp)
        if st.button("👍", key="gemini-up"):
            send_to_rlhf(prompt, chatgpt_resp, gemini_resp, preferred="gemini")