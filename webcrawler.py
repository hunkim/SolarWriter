import streamlit as st
from crawl4ai import AsyncWebCrawler
from typing import Tuple, List

async def get_web_content(url: str) -> Tuple[str, List[str]]:
    """Fetch website content and return both content and found URLs.
    
    Args:
        url: The URL to crawl
        
    Returns:
        Tuple containing (markdown_content, list_of_links)
    """
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            # Check if the page exists/is valid
            if not result or not result.markdown:
                st.warning(f"Page not found or empty: {url}")
                return "", []
            
            return result.markdown, result.links
    except Exception as e:
        st.error(f"Error fetching website: {e}")
        return "", [] 