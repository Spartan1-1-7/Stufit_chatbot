import streamlit as st
import random
import time
from styles.styling import load_css, set_page_config, load_avatar_css
from LLM_model import generate_response

# Configure page and load styles
set_page_config()
load_css()
load_avatar_css()

# Streamed response emulator
def response_generator(prompt):
    response = generate_response(prompt)
    for word in response.split():
        yield word + " "
        time.sleep(0.1)

# Main title
st.markdown('<h1 class="main-title">Stufit Report Analyzer</h1>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="media/User_pfp.jpg"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="media/stufit_logo.png"):
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user", avatar="media/User_pfp.jpg"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="media/stufit_logo.png"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

