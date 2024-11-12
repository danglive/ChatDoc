# app.py

import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from utils import (
    process_pdf,
    create_chroma_store,
    load_chroma_store,
    qa_ret
)
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load environment variables early in the application
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    """
    Endpoint to upload a PDF file and process it.

    Args:
        file (UploadFile): The uploaded PDF file.

    Returns:
        dict: A message indicating success or failure.
    """
    try:
        print("Received file:", file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file_content = await file.read()
            temp_file.write(file_content)
            temp_file_path = temp_file.name
            print("Temporary file path:", temp_file_path)

        document_chunks = process_pdf(temp_file_path)
        print("Number of document chunks:", len(document_chunks))

        # Load the embeddings model with trust_remote_code=True
        embedding_model = HuggingFaceEmbeddings(
            model_name="Lajavaness/bilingual-embedding-small",
            model_kwargs={"trust_remote_code": True}
        )
        print("Embedding model initialized.")

        create_chroma_store(document_chunks, embedding_model)
        print("Chroma store created and persisted.")

        os.remove(temp_file_path)
        print("Temporary file removed.")

        return {"message": "PDF successfully processed and stored in vector DB"}

    except Exception as e:
        print("An error occurred in upload_pdf:", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/ask-question/")
async def ask_question(question_request: QuestionRequest) -> dict:
    """
    Endpoint to ask a question and retrieve a response.

    Args:
        question_request (QuestionRequest): The question payload.

    Returns:
        dict: The answer to the question.
    """
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="Lajavaness/bilingual-embedding-small",
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = load_chroma_store(embedding_model)

        question = question_request.question
        response = qa_ret(vector_store, question)

        return {"answer": response}

    except Exception as e:
        print("An error occurred in ask_question:", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {str(e)}")


@app.get("/")
async def health_check() -> dict:
    """
    Endpoint to check the health of the application.

    Returns:
        dict: The status of the application.
    """
    return {"status": "Success"}