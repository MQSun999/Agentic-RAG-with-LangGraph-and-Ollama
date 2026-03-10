from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from docling.document_converter import DocumentConverter
import hashlib
from pathlib import Path


load_dotenv("./../.env")
# configurations
DATA_DIR = "data"
CHROMA_DIR = "./chroma_financial_db"
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = 'nomic-embed-text'
BASE_URL = 'http://localhost:11434'
NUM_CTX = 8192 # 向量化维度


# 从文件名提取关键信息，将关键信息保存为字典
def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extract metadata from filename.
    
    Expected format: {company} {doc_type} {quarter} {year}.pdf
    Examples:
    - amazon 10-k 2024.pdf

    
    Returns:
        dict with company_name, doc_type, fiscal_year, fiscal_quarter
    """

    name = filename.replace('.pdf', '')   
    parts = name.split()

    metadata = {}
    if len(parts) == 4:
        metadata['fiscal_quarter'] = parts[2]
        metadata['fiscal_year'] = int(parts[3])

    else:
        metadata['fiscal_quarter'] = None
        metadata['fiscal_year'] = int(parts[2])

    metadata['company_name'] = parts[0]
    metadata['doc_type'] = parts[1]

    return metadata

# 使用docling提取pdf中的数据，按页分隔
def extract_pdf_pages(pdf_path):

    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    page_break = "<!-- page break -->"

    markdown_text = result.document.export_to_markdown(page_break_placeholder=page_break)

    pages = markdown_text.split(page_break)

    return pages

# 计算一整个文件内容的SHA-256哈希值，为了之后数据注入时的去重
def compute_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# 将所有页面的向量化注入数据库
def ingest_docs_in_vectordb(pdf_path:Path, processed_hashes:set, vector_store:Chroma):
    print(f"Processing: {pdf_path.name}")

    file_hash = compute_file_hash(pdf_path)
    if file_hash in processed_hashes:
        print(f"[SKIP] already processed: {pdf_path}")
        return 

    pages = extract_pdf_pages(pdf_path)

    file_metadata = extract_metadata_from_filename(pdf_path.name)

    processed_pages = []

    for page_num, page_text in enumerate(pages, start=1):
        metadata_dict = file_metadata.copy()
        metadata_dict['page'] = page_num
        metadata_dict['file_hash'] = file_hash
        metadata_dict['source_file'] = pdf_path.name

        doc = Document(page_content=page_text, metadata=metadata_dict)

        processed_pages.append(doc)

    
    vector_store.add_documents(documents=processed_pages)