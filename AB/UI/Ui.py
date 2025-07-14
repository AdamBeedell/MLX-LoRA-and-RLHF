## Import stuff

import streamlit as st
import torch


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



col1, col2, col3 = st.columns(3)

with col1:
    st.chat_input()

#with col2:

with col3:
    if st.button("Add to RLHF dataset"):
        st.info("Added Maybe!")