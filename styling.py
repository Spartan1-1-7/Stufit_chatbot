import streamlit as st
import os

def load_css():
    """Load CSS styles from external file"""
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    css_file = os.path.join(current_dir, "styles.css")
    
    # Read the CSS file
    with open(css_file, "r") as f:
        css_content = f.read()
    
    # Apply the CSS
    st.markdown(f"""
    <style>
    {css_content}
    </style>
    """, unsafe_allow_html=True)

def load_avatar_css():
    """Load additional avatar styling"""
    st.markdown("""
    <style>
    /* Avatar styling for chat messages */
    .stChatMessage img {
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        object-fit: cover !important;
        border: 2px solid #FF6B6B !important;
        margin-right: 10px !important;
    }
    
    /* Chat message container with avatar */
    .stChatMessage {
        display: flex !important;
        align-items: flex-start !important;
        gap: 10px !important;
    }
    
    /* Message content styling */
    .stChatMessage > div:last-child {
        flex: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def set_page_config():
    """Set page configuration"""
    st.set_page_config(
        page_title="Simple Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
