
import streamlit as st
import torch
import torch.nn as NN
import torch.nn.functional as F
from openai import OpenAI
import anthropic
import google.generativeai as genai
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle


# Load secrets
client = OpenAI(api_key=st.secrets["providers"]["openai_key"])
anthropic_client = anthropic.Anthropic(api_key=st.secrets["providers"]["anthropic_key"])
genai.configure(api_key=st.secrets["providers"]["gemini_key"])
MISTRAL_KEY = st.secrets["providers"]["mixtral_key"]

# ----- Styling -----
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #09091a, #151596, #911363, #2a043b);
        background-size: 400% 400%;
        animation: gradientShift 25s ease infinite;
        color: white;
    }

    .glow-text {
        display: inline-block;
        animation: pulseGlow 1s ease-in-out infinite;
        font-size: 4em;
        font-weight: bold;
        text-align: center;
        width: 100%;
    }
    </style>
    <h1 class="glow-text">MoMoE</h1>
""", unsafe_allow_html=True)

# ----- Dummy Model -----
class MoMoEClassifier(NN.Module):   ### This creates a class for our specific NN, inheriting from the pytorch equivalent
    def __init__(self):  
        super().__init__()  ## super goes up one level to the torch NN module, and initializes the net
        self.fc1 = NN.Linear(512, 256)  ############################################################################ Figure out a good context window here
        self.fc2 = NN.Linear(256, 128)  # half as many nodes
        self.fc3 = NN.Linear(128, 64)   # half as many nodes
        self.fc4 = NN.Linear(64, 5) # Output layer (64 -> 5, one for each valid expert in the MoMoE model)

    def forward(self, x):  # feed forward
        x = F.relu(self.fc1(x))  # Normalization function (ReLU)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here, end of the road ("cross-entropy expects raw logits" - which are produced here, the logits will be converted to probabilities later by the cross-entropy function during training and softmax during training and inference)
        return x
    
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except:
    print(os.getcwd())


@st.cache_resource
def load_model():
    model = MoMoEClassifier()
    try:
        model.load_state_dict(torch.load("MoMoEWeights.pth", map_location="cpu"))
    except:
        print(os.getcwd())
    model.eval()
    return model

model = load_model()

# --- Vectorize Text (TF-IDF) ---

def tokenize(prompt):
    vec = vectorizer.transform([prompt]).toarray()
    return torch.tensor(vec, dtype=torch.float32)


# ----- Provider Dispatch -----
def dispatch_to_provider(provider_id, chat_history):
    if provider_id == 0:
        return query_chatgpt(chat_history)
    elif provider_id == 1:
        return query_gemini(chat_history)
    elif provider_id == 2:
        return query_mixtral(chat_history)
    elif provider_id == 3:
        return query_claude(chat_history)
    elif provider_id == 4:
        return "Math expert response (dummy)"
    else:
        return "Unknown provider"
    

# ----- Provider Map -----
ProviderLookup = {
    0: "ChatGPT",
    1: "GeminiLargeContext",
    2: "MixtralFrench",
    3: "ClaudeCode",
    4: "AceMath"
}

def gemini_len(history):
    text = "".join(m["content"] for m in history)
    return len(text) > 1000  # Adjust as needed; ~4 chars per token
    

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

def query_claude(history):
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in {"user", "assistant"}
    ]

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=messages
    )
    return response.content[0].text

def query_gemini(history):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history
    )
    response = model.generate_content(prompt)
    return response.text

def query_mixtral(history):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": m["role"], "content": m["content"]} for m in history]

    payload = {
        "model": "mistral-small-latest",  # or mistral-small-latest / mixtral-8x7b
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def get_provider_avatar(pid):
    avatar_lookup = {
        0: "☸😮",     # OpenAI
        1: "🌈",     # Gemini
        2: "🥖",     # Mixtral
        3: "🤖",     # Claude
    }
    return avatar_lookup.get(pid, "🤖")


# ----- Main Chat Logic -----
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message here...")

# User input received
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ✅ Gemini override for long inputs
    if gemini_len(st.session_state.chat_history):
        provider_id = 1  # Gemini
    else:
        # Classify normally
        token_tensor = tokenize(user_input)
        with torch.no_grad():
            logits = model(token_tensor)
            provider_id = torch.argmax(logits).item()            
    # Dispatch
    reply = dispatch_to_provider(provider_id, st.session_state.chat_history)
    st.session_state.chat_history.append({
    "role": "assistant",
    "content": reply,
    "provider_id": provider_id
    })


# Display chat
for i, msg in enumerate(st.session_state.chat_history):
    # Use custom avatar only for assistant responses
    if msg["role"] == "assistant":
        provider_id = msg.get("provider_id", None)
        avatar = get_provider_avatar(provider_id)
        with st.chat_message("assistant", avatar=avatar):
            st.write(msg["content"])
    else:
        with st.chat_message("user"):
            st.write(msg["content"])


