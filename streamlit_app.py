# streamlit_app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables early
load_dotenv()

# Set TOKENIZERS_PARALLELISM to false to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize session state for chat history and API URL
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'api_url' not in st.session_state:
    st.session_state['api_url'] = os.getenv("API_URL", "http://localhost:8000")

# Set Streamlit page configuration
st.set_page_config(
    page_title="ChatDoc",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define CSS styles
st.markdown("""
    <style>
    .flex-container {
        display: flex;
        flex-direction: column;
        padding: 10px;
        overflow-y: auto;
        max-height: 500px;
    }
    .align-right {
        justify-content: flex-end;
    }
    .align-left {
        justify-content: flex-start;
    }
    .user-message, .assistant-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 70%;
    }
    .user-message {
        background-color: #DCF8C6;
        align-self: flex-end;
    }
    .assistant-message {
        background-color: #E8E8E8;
        align-self: flex-start;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("Configuration")

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

api_url_input = st.sidebar.text_input(
    "API Base URL",
    value=st.session_state['api_url'],
    help="Base URL of the FastAPI backend, e.g., http://localhost:8000"
)

if is_valid_url(api_url_input):
    st.session_state['api_url'] = api_url_input
else:
    st.sidebar.error("Invalid API URL provided.")

# Function to upload PDF
def upload_pdf(file):
    """Uploads a PDF file to the /upload-pdf/ endpoint."""
    api_url = st.session_state['api_url']
    try:
        with st.spinner("Uploading PDF..."):
            files = {"file": (file.name, file, "application/pdf")}
            response = requests.post(f"{api_url}/upload-pdf/", files=files, timeout=60)
        if response.status_code == 200:
            st.sidebar.success("PDF successfully uploaded and processed.")
            logger.info("PDF uploaded successfully.")
            return True
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.sidebar.error(f"Failed to upload PDF: {error_detail}")
            logger.error(f"Failed to upload PDF: {error_detail}")
            return False
    except requests.exceptions.Timeout:
        st.sidebar.error("Request timed out. Please try again.")
        logger.error("PDF upload timed out.")
        return False
    except Exception as e:
        st.sidebar.error(f"An error occurred: {str(e)}")
        logger.error(f"An error occurred during PDF upload: {str(e)}", exc_info=True)
        return False

# Function to send question
def send_question(question):
    """Sends a question to the /ask-question/ endpoint and returns the answer."""
    api_url = st.session_state['api_url']
    try:
        with st.spinner("Fetching answer..."):
            payload = {"question": question}
            response = requests.post(f"{api_url}/ask-question/", json=payload, timeout=30)
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer returned.")
            logger.info("Received answer from API.")
            return answer
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Failed to retrieve answer: {error_detail}")
            logger.error(f"Failed to retrieve answer: {error_detail}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        logger.error("Question request timed out.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"An error occurred during question retrieval: {str(e)}", exc_info=True)
        return None

# Function to render chat history
def render_chat():
    """Renders the chat history in a chat-like interface."""
    for chat in st.session_state['chat_history']:
        if chat['role'] == 'user':
            st.markdown(
                f"""
                <div class="flex-container align-right">
                    <div class="user-message"> {chat['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif chat['role'] == 'assistant':
            st.markdown(
                f"""
                <div class="flex-container align-left">
                    <div class="assistant-message"> {chat['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Main Interface

st.title("📄 ChatDoc: Your PDF Question-Answering Assistant")

# Input form for sending a new question
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your question about the uploaded document:", "")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    # Append user message to chat history
    st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
    
    # Get the assistant's response
    answer = send_question(user_input)
    
    if answer:
        # Append assistant's response to chat history
        st.session_state['chat_history'].append({'role': 'assistant', 'content': answer})
        # Streamlit automatically reruns, so no need for st.rerun()

# Chat Interface
st.header("💬 Chat with Your Document")

# Chat container with scroll
st.markdown('<div id="chat-container" class="flex-container">', unsafe_allow_html=True)
render_chat()
st.markdown('</div>', unsafe_allow_html=True)

# Automatically scroll to the bottom of the chat
st.markdown("""
    <script>
    const chatContainer = window.parent.document.getElementById('chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    </script>
    """, unsafe_allow_html=True)

st.markdown("---")

# Sidebar: Upload PDF Document
st.sidebar.header("📥 Upload PDF Document")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Drag and drop your PDF file here. Limit: 200MB per file."
)

if uploaded_file is not None:
    if st.sidebar.button("Upload"):
        success = upload_pdf(uploaded_file)
        if success:
            st.balloons()

st.sidebar.markdown("---")

# Sidebar: Health Check
st.sidebar.header("🔍 Health Check")

if st.sidebar.button("Check Application Status"):
    try:
        api_url = st.session_state['api_url']
        response = requests.get(f"{api_url}/", timeout=10)
        if response.status_code == 200:
            status = response.json().get("status", "Unknown status")
            st.sidebar.success(f"Application Status: {status}")
            logger.info("Health check successful.")
        else:
            st.sidebar.error(f"Failed to retrieve status: {response.status_code}")
            logger.error(f"Failed to retrieve status: {response.status_code}")
    except requests.exceptions.Timeout:
        st.sidebar.error("Health check request timed out.")
        logger.error("Health check request timed out.")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {str(e)}")
        logger.error(f"An error occurred during health check: {str(e)}", exc_info=True)