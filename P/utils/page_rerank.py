from utils.schemas import ChunkMetadata, RankingKeywords
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
import re
from rank_bm25 import BM25Plus

CHROMA_DIR = "chroma_financial_db"
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "nomic-embed-text"
BASE_URL = "http://localhost:11434"
MODEL = "qwen3.5:397b-cloud"

llm = ChatOllama(base_url=BASE_URL, model=MODEL)

# 调用llm生成与问题可能相关的keyword，便于之后提取内容进行排序
def generate_ranking_keywords(user_query: str) -> list:
    # ALT + Z
    prompt = f"""Generate EXACTLY 5 financial keywords from SEC filings terminology.

                USER QUERY: {user_query}

                USE EXACT TERMS FROM 10-K/10-Q FILINGS:

                STATEMENT HEADINGS:
                "consolidated statements of operations", "consolidated balance sheets", "consolidated statements of cash flows", "consolidated statements of stockholders equity"

                INCOME STATEMENT:
                "revenue", "net revenue", "cost of revenue", "gross profit", "operating income", "net income", "earnings per share"

                BALANCE SHEET:
                "total assets", "cash and cash equivalents", "total liabilities", "stockholders equity", "working capital", "long-term debt"

                CASH FLOWS:
                "cash flows from operating activities", "net cash provided by operating activities", "cash flows from investing activities", "free cash flow", "capital expenditures"

                RULES:
                - Return EXACTLY 5 keywords
                - Use exact phrases from SEC filings
                - Match query topic (revenue -> revenue terms, cash -> cash flow terms)
                - Use "cash flows" (plural), "stockholders equity"

                EXAMPLES:
                "revenue analysis" -> ["revenue", "net revenue", "total revenue", "consolidated statements of operations", "net sales"]
                "cash flow performance" -> ["consolidated statements of cash flows", "cash flows from operating activities", "net cash provided by operating activities", "free cash flow", "operating activities"]
                "balance sheet strength" -> ["consolidated balance sheets", "total assets", "stockholders equity", "cash and cash equivalents", "long-term debt"]

                Generate EXACTLY 5 keywords:
                """
    
    llm_structured = llm.with_structured_output(RankingKeywords)
    result = llm_structured.invoke(prompt)

    return result.keywords

# 根据观察，大部分财务报告都是小标题和小标题后面衔接的一段比较有信息量，这个函数提取这些内容
def extract_headings_with_content(text:str) -> list:
    """
    Extract markdown headings with one paragraph of content after them.
    
    Args:
        text: Document text content
    
    Returns:
        List of extracted heading + content chunks
    """

    chunks = []

    sections = text.split('\n\n')

    i = 0
    while i< len(sections):
        section = sections[i].strip()

        pattern = r"^#+\s+"
        if re.match(pattern, section):
            heading = section

            # Get the next paragraph/content after the heading
            if i+1 < len(sections):
                next_content = sections[i+1].strip()

                chunk = f"{heading}\n\n{next_content}"
                i = i + 2
            
            else:
                chunk = heading
                i = i + 1

            chunks.append(chunk)

        else:
            i = i + 1

    return chunks

# 使用llm生成的keyword对从数据库检索出的每个页面进行BM25PLUS排序
def rank_documents_by_keywords(docs:list, keywords:list, k=5)-> list:
    """
    Rank documents using BM25Plus on heading+content chunks.
    
    Args:
        docs: List of Document objects to rank
        keywords: List of keywords to rank by
        k: Number of top documents to return
    
    Returns:
        List of top-k Document objects sorted by BM25 score
    """

    if not docs or not keywords:
        print("Either No doc or keywords found!")
        return docs
    
    query_tokens = " ".join(keywords).lower().split(" ")

    # extract chunks for each document
    doc_chunks = []
    for doc in docs:
        chunks = extract_headings_with_content(doc.page_content)

        combined = " ".join(chunks) if chunks else doc.page_content

        doc_chunks.append(combined.lower().split(' '))

    # Rank using BM25Plust

    bm25 = BM25Plus(doc_chunks)
    doc_scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)

    for rank, idx in enumerate(ranked_indices[:k], 1):
        print(f"   [{rank}] Doc {idx}: score={doc_scores[idx]:.4f}")


    return [docs[i] for i in ranked_indices[:k]]