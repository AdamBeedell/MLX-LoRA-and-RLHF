import streamlit as st
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

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
    <h1 class="glow-text">Andrei's SFT/PPO Model</h1>
""", unsafe_allow_html=True)

# ----- Inference Model -----
class InfModel():
    def __init__(self, tokenizer, model, max_post_length=500):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.max_post_length = max_post_length

    def __call__(self, text):
        inputs = self.tokenizer([text + ' <|endoftext|>'], truncation=True, max_length=self.max_post_length, padding="max_length", return_tensors='pt') 
        output = self.model.generate(**inputs, max_new_tokens=50, do_sample=True, top_k=10)
        output = output.squeeze(0)[self.max_post_length:]
        return self.tokenizer.decode(output, skip_special_tokens=True)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained("ayzor/qwen_summary_generator_ppo")
    return InfModel(tokenizer, model)

summarize = load_model()

# ----- Chat Logic -----
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    summary = summarize(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": summary})

# ----- Display Chat -----
for msg in st.session_state.chat_history:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🧪"):
            st.write(msg["content"])
    else:
        with st.chat_message("user"):
            st.write(msg["content"])



# text = "My wife brought home approximately 30 uniforms that needed to be repaired, washed, and returned. The uniforms were fixed, but were not washed commercially, but instead in my personal household washer (my wife didn't want to take the risk of damaging the uniforms by washing them commercially) and hung on a large rolling costume rack to drip-dry. Unfortunately, due to both of our schedules, the uniforms didn't get returned immediately, but sat in my house for several weeks. As my wife had keys to the band room, I left it to her good graces to take the uniforms back."
# summary = summarize(text)
# print(summary)
