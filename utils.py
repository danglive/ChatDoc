# utils.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Updated import

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Load environment variables as early as possible
load_dotenv()

# Set TOKENIZERS_PARALLELISM to false to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def process_pdf(pdf_path: str) -> list:
    """
    Process the PDF file and split it into smaller text chunks.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of document chunks.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    document_text = "".join(page.page_content for page in pages)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=40
    )
    chunks = text_splitter.create_documents([document_text])

    return chunks


def create_chroma_store(documents: list, embedding_model) -> Chroma:
    """
    Create and persist the Chroma vector store from document chunks.

    Args:
        documents (list): List of document chunks.
        embedding_model: Embedding model to use.

    Returns:
        Chroma: The persisted Chroma vector store.
    """
    vectorstore = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory="chroma_db"
    )
    # Removed vectorstore.persist() as it's deprecated
    return vectorstore


def load_chroma_store(embedding_model) -> Chroma:
    """
    Load the Chroma vector store from disk.

    Args:
        embedding_model: Embedding model to use.

    Returns:
        Chroma: The loaded Chroma vector store.
    """
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory="chroma_db"
    )
    return vectorstore


def qa_ret(vector_store, input_query: str) -> str:
    """
    Perform question answering using the vector store and input query.

    Args:
        vector_store: The Chroma vector store.
        input_query (str): The user's question.

    Returns:
        str: The answer to the question or an error message.
    """
    try:
        template = """
Instructions:
    You are trained to extract answers from the given Context and the User's Question. Your response must be based on semantic understanding, which means even if the wording is not an exact match, infer the closest possible meaning from the Context.

    Key Points to Follow:
    - Precise Answer Length: The answer must be between a minimum of 40 words and a maximum of 100 words.
    - Strict Answering Rules: Do not include any unnecessary text. The answer should be concise and focused directly on the question.
    - Professional Language: Do not use any abusive or prohibited language. Always respond in a polite and gentle tone.
    - No Personal Information Requests: Do not ask for personal information from the user at any point.
    - Concise & Understandable: Provide the most concise, clear, and understandable answer possible.
    - Semantic Similarity: If exact wording isn’t available in the Context, use your semantic understanding to infer the answer. If there are semantically related phrases, use them to generate a precise response. Use natural language understanding to interpret closely related words or concepts.
    - Unavailable Information: If the answer is genuinely not found in the Context, politely apologize and inform the user that the specific information is not available in the provided context.

Context:
{context}

User's Question: {question}

Respond in a polite, professional, and concise manner.
"""
        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )

        model = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=512
        )

        output_parser = StrOutputParser()

        rag_chain = setup_and_retrieval | prompt | model | output_parser
        response = rag_chain.invoke(input_query)
        return response

    except Exception as ex:
        return f"Error: {str(ex)}"