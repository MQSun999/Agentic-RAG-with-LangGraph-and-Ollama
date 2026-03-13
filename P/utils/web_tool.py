# tavily search dev serper, ddgs
from ddgs import DDGS
from langchain_community.tools import tool
from dotenv import load_dotenv
load_dotenv()


@tool
def web_search(query: str, num_results: int = 10) -> str:
    """Use this tool whenever you need to access realtime or latest information.
        Search the web using DuckDuckGo.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)

    Returns:
        Formatted search results with titles, descriptions, and URLs
    """

    results = DDGS().text(query=query, max_results=num_results, region="us-en")

    if not results:
        return f"No results found for '{query}'"

    formatted_results = [f"Search results for search query: '{query}'"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        href = result.get("href", "")
        body = result.get("body", "No description available")

        text = f"{i}. **{title}**\n   {body}\n   {href}"

        formatted_results.append(text)

    return "\n\n".join(formatted_results)
