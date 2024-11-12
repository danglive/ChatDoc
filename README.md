# ChatDoc

ChatDoc is a Python-based application that allows users to upload PDF documents, processes them into manageable chunks, and enables question-answering against the content using OpenAI's language models. The application utilizes FastAPI for the backend API and provides a Streamlit-based user interface for ease of use.

## Features

- **PDF Upload and Processing**: Upload PDF documents which are then processed and split into smaller text chunks.
- **Vector Store Creation**: Creates a Chroma vector store from the processed document chunks using HuggingFace embeddings.
- **Question Answering**: Ask questions about the uploaded documents and receive concise, accurate answers based on the document content.
- **Streamlit UI**: A user-friendly interface built with Streamlit for interacting with the application.

## Technologies Used

- **FastAPI**: For building the backend API.
- **LangChain**: For handling language model interactions.
- **Chroma**: For vector storage.
- **HuggingFace Embeddings**: For generating document embeddings.
- **OpenAI GPT-3.5-turbo or 4o-mini**: For the language model used in question answering.
- **Streamlit**: For the frontend user interface.
- **Other Libraries**: PyPDF, dotenv, etc.

## Installation

### Prerequisites

- Python 3.9+
- Git

### Clone the Repository

```bash
git clone https://github.com/danglive/chatdoc.git
cd chatdoc
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Configure Environment Variables

Copy the example .env file and edit it with your actual OpenAI API key:
```bash
cp .env.example .env
```
Open .env and add your OPENAI_API_KEY:
```bash
# .env
OPENAI_API_KEY=your_actual_openai_api_key_here
API_URL=http://localhost:8000
TOKENIZERS_PARALLELISM=false
```

### Run the Application

Start the FastAPI Server
```bash
uvicorn app:app --reload
```

Run the Streamlit App

In a separate terminal (with the virtual environment activated), run:
```bash
streamlit run streamlit_app.py
```