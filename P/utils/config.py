from dotenv import load_dotenv
import os
from typing import Any
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from langchain_community.utilities import SQLDatabase

BASE_URL = "http://localhost:11434"

# LLM configuration
RAG_LLM = "gpt-oss:20b-cloud"
WEB_LLM = "gpt-oss:20b-cloud"
SQL_LLM = "gpt-oss:20b-cloud"
AGENT_LLM = "gpt-oss:120b-cloud"

def get_llm( model: str, base_url: str = BASE_URL, **kwargs: Any) -> ChatOllama:
    """
    Create a ChatOllama instance with support for custom parameters.

    Args:
        base_url: Ollama service URL
        model: Model name
        **kwargs: Additional parameters supported by ChatOllama, such as temperature, reasoning, etc.

    Returns:
        ChatOllama instance

    Examples:
        # Basic usage
        llm = get_llm(BASE_URL, RAG_LLM)

        # Custom parameters
        llm = get_llm(
            BASE_URL,
            RAG_LLM,
            temperature=0.7,
            top_p=0.9,
            reasoning="enabled"
        )
    """
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        **kwargs
    )
    return llm

# Vector Database Configuration
CHROMA_DIR = Path(
    "E:/Udemy/Agentic-RAG-with-LangGraph-and-Ollama/P/chroma_financial_db"
)
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL, num_ctx=8192)

def get_vector_store() -> Chroma:
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vector_store

# SQL Database Configuration
SQL_PATH = Path(
    r"E:\Udemy\Agentic-RAG-with-LangGraph-and-Ollama\P\db\employees_db-full-1.0.6.db"
)

def get_db_connection():
    db = SQLDatabase.from_uri(f"sqlite:///{SQL_PATH}")
    return db

