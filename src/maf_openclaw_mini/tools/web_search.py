"""
Web Search Tool

Provides web search capability for the agent.

Following Microsoft Agent Framework patterns:
- https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-tools

Note: MAF provides HostedWebSearchTool for Azure-hosted search.
This implementation provides a fallback using DuckDuckGo for local use.
"""

import os
import re
from typing import Annotated, Optional
from pydantic import Field
from agent_framework import tool
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


@tool(
    name="web_search",
    description="Search the web for current information. Use this when you need up-to-date information that might not be in your training data."
)
async def web_search(
    query: Annotated[str, Field(description="The search query")],
    num_results: Annotated[int, Field(description="Number of results to return")] = 5,
) -> str:
    """
    Search the web using DuckDuckGo HTML search.

    This is a fallback implementation when HostedWebSearchTool is not available.
    For production use with Azure, consider using HostedWebSearchTool instead.

    Args:
        query: Search query
        num_results: Number of results to return (max 10)

    Returns:
        Formatted search results
    """
    num_results = min(num_results, 10)

    try:
        # Use DuckDuckGo HTML search
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                url,
                data={"q": query},
                headers=headers,
                follow_redirects=True,
            )

            if response.status_code != 200:
                return f"Search failed with status {response.status_code}"

            # Parse results
            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            # Find result links
            for result in soup.select(".result"):
                if len(results) >= num_results:
                    break

                title_elem = result.select_one(".result__title")
                snippet_elem = result.select_one(".result__snippet")
                link_elem = result.select_one(".result__url")

                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True)
                    link = link_elem.get_text(strip=True) if link_elem else ""

                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": link,
                    })

            if not results:
                return f"No results found for '{query}'"

            # Format results
            formatted = [f"Search results for '{query}':\n"]
            for i, r in enumerate(results, 1):
                formatted.append(f"{i}. **{r['title']}**")
                formatted.append(f"   {r['snippet']}")
                if r['url']:
                    formatted.append(f"   URL: {r['url']}")
                formatted.append("")

            return "\n".join(formatted)

    except httpx.TimeoutException:
        return "Search timed out. Please try again."
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool(
    name="fetch_url",
    description="Fetch and extract content from a URL. Use this to read the contents of a web page."
)
async def fetch_url(
    url: Annotated[str, Field(description="The URL to fetch (must be http:// or https://)")],
    extract_type: Annotated[str, Field(description="What to extract: 'text', 'title', or 'summary'")] = "text",
    max_length: Annotated[int, Field(description="Maximum characters to return")] = 4000,
) -> str:
    """
    Fetch and extract content from a URL.

    Args:
        url: The URL to fetch
        extract_type: 'text' (full text), 'title' (just title), or 'summary' (first paragraphs)
        max_length: Maximum characters to return

    Returns:
        Extracted content
    """
    # Validate URL
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    # Block internal/private URLs
    private_patterns = [
        r"^https?://localhost",
        r"^https?://127\.",
        r"^https?://10\.",
        r"^https?://172\.(1[6-9]|2[0-9]|3[01])\.",
        r"^https?://192\.168\.",
        r"^https?://\[::1\]",
    ]

    for pattern in private_patterns:
        if re.match(pattern, url):
            return "Error: Cannot fetch private/internal URLs"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                url,
                headers=headers,
                follow_redirects=True,
            )

            if response.status_code != 200:
                return f"Failed to fetch URL (status {response.status_code})"

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            # Extract based on type
            if extract_type == "title":
                title = soup.title.string if soup.title else "No title found"
                return f"Title: {title}"

            elif extract_type == "summary":
                # Get first few paragraphs
                paragraphs = soup.find_all("p")
                text_parts = []
                char_count = 0

                for p in paragraphs[:5]:
                    text = p.get_text(strip=True)
                    if text and len(text) > 50:
                        text_parts.append(text)
                        char_count += len(text)
                        if char_count > max_length:
                            break

                if not text_parts:
                    return "No summary content found"

                title = soup.title.string if soup.title else "Untitled"
                return f"**{title}**\n\n" + "\n\n".join(text_parts)[:max_length]

            else:  # text
                # Get full text content
                text = soup.get_text(separator="\n", strip=True)

                # Clean up excessive whitespace
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = text[:max_length]

                title = soup.title.string if soup.title else "Untitled"
                return f"**{title}**\n\n{text}"

    except httpx.TimeoutException:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Failed to fetch URL: {str(e)}"
