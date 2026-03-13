from langgraph.graph import START, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils import schemas, config, rag_tool, sql_tool, web_tool
from dotenv import load_dotenv
import os
from langchain.agents import create_agent

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

llm = config.get_llm(model=config.AGENT_LLM, reasoning = False)

# =============================================================================
# Helper Function
# =============================================================================
def get_latest_user_query(messages: list):

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content

    return messages[0].content if messages else ""

# =============================================================================
# RAG Nodes
# =============================================================================


# Retrieve documents based on user query
def retrieve_node(state):

    print("[RETRIEVE] fetching documents...")

    query = get_latest_user_query(state["messages"])

    rewritten_queries = state.get("rewritten_queries", [])

    # use rewriten queries if present
    queries_to_search = rewritten_queries if rewritten_queries else [query]

    all_results = []
    for idx, search_query in enumerate(queries_to_search, 1):
        print(f"[RETRIEVE] Query {idx}: {search_query}")

        # 3(Reranking) -> 3*10(Retrieval) -> 3*10*20 (MMR)
        result = rag_tool.page_retrieval_rerank.invoke({"query": search_query, "k": 3})

        text = f"## Query {idx}: {search_query}\n\n### Retrieved Documents:\n{result}"
        all_results.append(text)

    combined_result = "\n\n".join(all_results)

    os.makedirs("debug_logs", exist_ok=True)
    with open("debug_logs/self_rag.md", "w", encoding="utf-8") as f:
        f.write(combined_result)

    return {"retrieved_docs": combined_result}


# Grade document relevance and filter out irrelevant ones
def grade_documents_node(state):

    print("[GRADE] Evaluating document relevance")

    query = get_latest_user_query(state["messages"])
    documents = state.get("retrieved_docs", "No document available!")

    llm_structured = llm.with_structured_output(schemas.GradeDocuments)

    system_prompt = """You are a grader assessing relevance of retrieved documents to a user query.

                It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

                If the document contains keyword(s) or semantic meaning related to the user query, grade it as relevant.

                Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the query.
                
                You MUST respond with ONLY a JSON object, no other text:
                {"binary_score": "yes"}
                or
                {"binary_score": "no"}
                """

    system_msg = SystemMessage(system_prompt)

    messages = [
        system_msg,
        HumanMessage(f"Retrieved Document: {documents}\n\nUser query: {query}"),
    ]

    response = llm_structured.invoke(messages)

    print(f"[GRADE] Relevance:  {response.binary_score}")

    if response.binary_score == "yes":
        return {"retrieved_docs": documents}

    else:
        return {"retrieved_docs": ""}


# Generate answer based on retrieved documents
def generate_node(state):
    print("[GENERATE] Creating Answer")

    query = get_latest_user_query(state["messages"])
    documents = state.get("retrieved_docs", "")

    system_prompt = """You are a financial document analyst providing detailed, accurate answers.

                OUTPUT FORMAT:
                Write a comprehensive answer (200-300 words) in MARKDOWN format:
                - Use ## headings for sections
                - Use **bold** for emphasis
                - Use bullet points or numbered lists
                - Include inline citations like [1], [2] where applicable

                GUIDELINES:
                - Base your answer ONLY on the provided documents
                - Be specific with numbers, dates, and metrics
                - If information is missing, acknowledge it
                - Use proper financial terminology

                CITATIONS:
                At the end, list references in this format:
                **References:**
                1. Company: x, Year: y, Quarter: z, Page: n"""

    query_prompt = f"Retrieved Document: {documents}\n\nUser query: {query}"

    system_msg = SystemMessage(system_prompt)
    user_msg = HumanMessage(query_prompt)

    messages = [system_msg, user_msg]

    response = llm.invoke(messages)

    os.makedirs("debug_logs", exist_ok=True)
    with open("debug_logs/self_rag_answer.md", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}")
        f.write(response.content)

    return {"messages": [response]}


# Transform the query to produce better search queries
def transform_query_node(state):

    query = get_latest_user_query(state["messages"])
    rewritten_queries = state.get("rewritten_queries", [])

    llm_structured = llm.with_structured_output(schemas.SearchQueries)

    system_prompt = """You are a query re-writer that decomposes complex queries into focused search queries optimized for vectorstore retrieval.

                DECOMPOSITION STRATEGY:
                Break down the original query into 1-3 specific, focused queries where each query targets:
                - A single company (e.g., "Amazon revenue 2023" vs "Google revenue 2023")
                - A specific time period (e.g., "Q1 2024" vs "Q2 2024")
                - A specific metric or aspect (e.g., "revenue" vs "net income")
                - A specific document section (e.g., "risk factors" vs "business overview")

                GUIDELINES:
                - Expand abbreviations (e.g., "rev" -> "revenue", "GOOGL" -> "Google")
                - Add financial context if missing
                - Make each query self-contained and specific
                - Keep queries concise but clear (5-10 words each)
                - Avoid repeating previously tried queries

                EXAMPLES:
                - "Compare Apple and Google revenue in 2024 Q1" → 
                ["Apple total revenue Q1 2024", "Google total revenue Q1 2024"]
                
                - "Amazon's revenue growth from 2022 to 2024" →
                ["Amazon revenue 2022", "Amazon revenue 2023", "Amazon revenue 2024"]
                
                - "What were the main risks for Microsoft in 2023?" →
                ["Microsoft risk factors 2023", "Microsoft business challenges 2023"]
                You MUST respond with ONLY a JSON object
                """

    query_context = f"Original Query: {query}"
    if rewritten_queries:
        query_context = (
            query_context
            + f"\n\n These queries have been already generated. do not generate same queries again.\n"
        )
        for idx, query in enumerate(rewritten_queries, 1):
            query_context = query_context + f"Query {idx}: {query}\n\n"

    query_context = (
        query_context
        + "\n\nGenerate 1-3 focused search queries that decompose the original query. Each query should target a specific aspect."
    )

    system_msg = SystemMessage(system_prompt)
    user_msg = HumanMessage(query_context)

    messages = [system_msg, user_msg]
    response = llm_structured.invoke(messages)

    new_queries = response.search_queries

    print(f"New Search Queries: {new_queries}")

    return {"rewritten_queries": new_queries}


# =============================================================================
# RAG Router Logic
# =============================================================================


# Decide whether to generate answer or transform query
def should_generate(state):
    print("[ROUTER] Assess graded documents")

    retrieved_docs = state.get("retrieved_docs", "")

    if not retrieved_docs or retrieved_docs.strip() == "":
        print(f"[ROUTER] No relevant documents - transforming query")
        return "transform_query"

    else:
        print("[ROUTER] Have relevant documents - generating answer")
        return "generate"


# Check for hallucinations and whether answer addresses query
def check_answer_quality(state):

    query = get_latest_user_query(state["messages"])
    documents = state.get("retrieved_docs", "")
    generation = state["messages"][-1].content

    llm_hallucinations = llm.with_structured_output(schemas.GradeHallucinations)

    hallucination_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
                           Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
                           You MUST respond with ONLY a JSON object, no other text:
                           {"binary_score": "yes"}
                           or
                           {"binary_score": "no"}
                            """

    system_msg = SystemMessage(hallucination_prompt)
    user_msg = HumanMessage(
        f"Set of facts:\n\n{documents}\n\nLLM Generation: {generation}"
    )

    messages = [system_msg, user_msg]
    response = llm_hallucinations.invoke(messages)

    hallucination_grade = response.binary_score

    # if result is grounded into the facts or retrieved docs
    if hallucination_grade == "yes":
        # now check answer quality
        print("[ROUTER] Generation is gounded in documents")

        print("[ROUTER] Checking answer quality")
        llm_answer = llm.with_structured_output(schemas.GradeAnswer)

        answer_prompt = """You are a grader assessing whether an answer addresses / resolves a query.

                        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the query.
                        
                        You MUST respond with ONLY a JSON object, no other text:
                           {"binary_score": "yes"}
                           or
                           {"binary_score": "no"}"""

        system_msg = SystemMessage(answer_prompt)

        user_msg = HumanMessage(f"User Query: {query}\n\n LLM Generation: {generation}")

        messages = [system_msg, user_msg]

        answer_response = llm_answer.invoke(messages)
        answer_grade = answer_response.binary_score

        if answer_grade == "yes":
            print("[ROUTER] generation is good. - USEFUL")
            return END
        else:
            print("[ROUTER] Generation does not address the query - NOT USEFUL")
            return "transform_query"

    else:
        print("[ROUTER] Generation NOT grounded in the response")
        return "generate"

# =============================================================================
# SQL Nodes
# =============================================================================

def sql_agent_node(state):

    query = get_latest_user_query(state["messages"])

    system_prompt = (
        system_prompt
    ) = f"""You are an expert SQL analyst working with an employees database.

                    Database Schema:
                    {sql_tool.SCHEMA}

                    Your workflow for answering questions:
                    1. Use `get_database_schema` first to understand available tables and columns (if needed)
                    2. Use `generate_sql_query` to create SQL based on the question
                    3. Use `execute_sql_query` to run the validated query
                    4. If there's an error, use `fix_sql_error` to correct it and try again (up to 3 times)

                    Rules:
                    - Always follow the workflow step by step
                    - If a query fails, use the fix tool and try again
                    - Provide clear, informative answers
                    - Be precise with table and column names
                    - Handle errors gracefully and try to fix them
                    - If you fail after 3 attempts, explain what went wrong

                    Available tools:
                    - get_database_schema: Get table structure info
                    - generate_sql_query: Create SQL from question
                    - execute_sql_query: Run the query
                    - fix_sql_error: Fix failed queries

                    Remember:
                    -Always validate queries before executing them for safety.
                    - Provide clear, formatted results in MARKDOWN
                    - Use tables for structured data"""

    agent = create_agent(
        model=llm, tools=sql_tool.ALL_SQL_TOOLS, system_prompt=system_prompt
    )

    result = agent.invoke({"messages": [HumanMessage(query)]})

    print(f"[SQL] Agent completed!")

    return {"messages": result["messages"]}


# =============================================================================
# WEB Search Nodes
# =============================================================================
def web_search_node(state):

    query = get_latest_user_query(state["messages"])

    system_prompt = """You are a helpful web search assistant. Be concise and accurate.

                    **Workflow:**
                    1. Use web_search tool to find relevant information
                    2. Synthesize information from multiple sources
                    3. Provide a clear, comprehensive answer

                    **OUTPUT FORMAT:**
                    Write a clear answer (150-250 words) in **MARKDOWN** format:
                    - Use ## headings for sections
                    - Use **bold** for emphasis
                    - Use bullet points for key facts

                    **GUIDELINES:**
                    - Base your answer on web search results
                    - Synthesize information from multiple sources
                    - Be factual and objective
                    
                    **REMEMBER:**
                    - Don't use web_search tool more than two times, if you can't find valid answer after two times tool call, 
                      tell user you have not searched relative answer on the web
                    """


    agent = create_agent(
        model=llm, tools=[web_tool.web_search], system_prompt=system_prompt
    )

    result = agent.invoke({"messages": [HumanMessage(query)]})

    print(f"[WEB] Agent is completed!")

    return {"messages": result["messages"]}