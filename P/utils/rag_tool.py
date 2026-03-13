from dotenv import load_dotenv
import os
from langchain_community.tools import tool
from .page_retrieval import extract_filters, search_docs
from .page_rerank import generate_ranking_keywords, rank_documents_by_keywords
from utils import config
load_dotenv()

llm = config.get_llm(model=config.RAG_LLM)
vector_store = config.get_vector_store()



@tool
def page_retrieval_rerank(query: str, k:int = 5):
    """
    Retrieve relevant financial documents from ChromaDB.
    Extracts filters from query and retrieves matching documents.

    Args:
        query: The search query (e.g., "What was Amazon's revenue in Q2 2025?")
        k: Number of documents to retrieve. generally prefer 5 docs

    Returns:
        Retrieved documents with metadata as formatted string
    """
    print(f"\n[TOOL] retrieve_docs called")
    print(f"[QUERY] {query}")

    # 必要关键词字典
    keywords = extract_filters(query, llm=llm)
    print(f"keywords: {keywords}")
    # 参考关键词字典
    references = generate_ranking_keywords(query, llm=llm)
    print(f"references: {references}")
    # 使用关键词搜索数据库内容
    raw_result = search_docs(
        query=query,
        vector_store=vector_store,
        filters=keywords,
        ranking_keywords=references,
        k=10 * k,
    )
    # 返回排序后的页面内容
    ranked_result = rank_documents_by_keywords(
        docs=raw_result, keywords=references, k=k
    )

    print(f"[RETRIEVED] {len(ranked_result)} documents")

    # format extracted docs or chunks
    if len(ranked_result) == 0:
        return f"No ducuments found for the query: '{query}'. Try rephrasing query or use different filter."

    # final format
    # --- Document {i} ---
    retrieved_text = []
    for i, doc in enumerate(ranked_result, 1):
        doc_text = [f"--- Document {i} ---"]

        # add all metadata
        for key, value in doc.metadata.items():
            doc_text.append(f"{key}: {value}")

        # add content
        doc_text.append(f"\nContent:\n{doc.page_content}")

        text = "\n".join(doc_text)
        retrieved_text.append(text)

    retrieved_text = "\n".join(retrieved_text)

    os.makedirs("debug_logs", exist_ok=True)
    with open("debug_logs/retrieved_reranked_docs.md", "w", encoding="utf-8") as f:
        f.write(retrieved_text)

    return retrieved_text