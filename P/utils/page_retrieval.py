from utils.schemas import ChunkMetadata, RankingKeywords
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# 调用llm从用户的问题中提取关键词，返回关键词字典
def extract_filters(user_query:str, llm) -> dict:

    llm_structured = llm.with_structured_output(ChunkMetadata)

    prompt = f"""Extract metadata filters from the query. Output valid JSON only. Use null (not None) for missing fields.

                USER QUERY: {user_query}

                COMPANY MAPPINGS:
                - Amazon/AMZN -> amazon
                - Google/Alphabet/GOOGL/GOOG -> google
                - Apple/AAPL -> apple
                - Microsoft/MSFT -> microsoft
                - Tesla/TSLA -> tesla
                - Nvidia/NVDA -> nvidia
                - Meta/Facebook/FB -> meta

                DOC TYPE:
                - Annual report -> 10-k
                - Quarterly report -> 10-q
                - Current report -> 8-k

                EXAMPLES:
                "Amazon Q3 2024 revenue" -> {{"company_name": "amazon", "doc_type": "10-q", "fiscal_year": 2024, "fiscal_quarter": "q3"}}
                "Apple 2023 annual report" -> {{"company_name": "apple", "doc_type": "10-k", "fiscal_year": 2023}}
                "Tesla profitability" -> {{"company_name": "tesla"}}

                Extract metadata:
                """
    
    metadata = llm_structured.invoke(prompt)
    filters = metadata.model_dump(exclude_none=True)

    return filters

# 生成搜索数据库的语法，为向量数据库的相似性搜索构建查询参数字典
def build_search_kwargs(filters, ranking_keywords, k=3) -> dict:

    search_kwargs = {"k": k, 'fetch_k': k*20}

    if filters:
        if len(filters) == 1:
            search_kwargs['filter'] = filters # filters = {'company_name': 'amazon'}

        else:
            # filters = {'company_name': 'amazon', 'doc_type':'10-k'}
            # [{'company_name': 'amazon'}, {'doc_type':'10-k'}]
            filters_conditions = [{k:v} for k, v in filters.items()]
            search_kwargs['filter'] = {"$and": filters_conditions}

    # Add document content filters using ranking keywords
    if ranking_keywords:
        if len(ranking_keywords) == 1:
            search_kwargs['where_document'] = {'$contains': ranking_keywords[0]}
        else:
            search_kwargs['where_document'] = {
                "$or": [{'$contains': keyword} for keyword in ranking_keywords]
            }

    return search_kwargs

# 搜索数据库内容
def search_docs(query:str, vector_store:Chroma, filters={}, ranking_keywords=[], k=3, ) -> list:
    """
        Search documents with metadata and content filters.
        
        Args:
            query (str): Search query text
            filters (dict): Metadata filters (e.g., {"company_name": "amazon", "fiscal_year": 2023})
            ranking_keywords (list): Keywords for content filtering (documents must contain at least one)
            k (int): Number of results (default: 5)
        
        Returns:
            list: Matching Document objects
        
        Example:
            docs = search_docs(
                query="Analyze cash flow",
                filters={"company_name": "amazon", "doc_type": "10-k"},
                ranking_keywords=["cash flow", "liquidity"],
                k=10
            )
        """
    search_kwargs = build_search_kwargs(filters, ranking_keywords, k)

    retriever = vector_store.as_retriever(
        search_type= "mmr",
        search_kwargs = search_kwargs
    )

    return retriever.invoke(query)

