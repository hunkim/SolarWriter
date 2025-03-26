import os
from typing import Dict, Any
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

def llm(keyword: str, prompt: str="", use_google_search: bool=True) -> Dict[str, Any]:
    """
    Perform a search using Google's Generative AI with optional Google Search integration
    
    Args:
        keyword: The search query
        prompt: Optional prompt to prepend to the query
        use_google_search: Whether to use Google Search tool (default: True)
        
    Returns:
        Dictionary containing summary and sources
    """
    # Initialize the Google Generative AI client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_id = "gemini-2.0-flash"
    model_id = "gemini-2.5-pro-exp-03-25"
    
    # Configure tools based on options
    tools = []
    if use_google_search:
        tools.append(Tool(google_search=GoogleSearch()))
    
    # Generate content
    response = client.models.generate_content(
        model=model_id,
        contents=prompt + keyword,
        config=GenerateContentConfig(
            tools=tools,
        ),
    )

    # Extract text from the first candidate's content
    if response.candidates and response.candidates[0].content.parts:
        text = response.candidates[0].content.parts[0].text
    else:
        raise Exception("No content found in response")

    # Extract sources from grounding metadata
    sources = []
    web_search_queries = []
    
    if use_google_search and hasattr(response.candidates[0], "grounding_metadata"):
        metadata = response.candidates[0].grounding_metadata
        web_search_queries = metadata.web_search_queries if hasattr(metadata, "web_search_queries") else []

        # Create a mapping of chunk indices to web sources
        web_sources = {}
        if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
            for i, chunk in enumerate(metadata.grounding_chunks):
                if hasattr(chunk, "web") and chunk.web:
                    web_sources[i] = {
                        "title": chunk.web.title,
                        "url": chunk.web.uri,
                        "contexts": [],
                    }

        # Add text segments to corresponding sources
        if hasattr(metadata, "grounding_supports") and metadata.grounding_supports:
            for support in metadata.grounding_supports:
                for chunk_idx in support.grounding_chunk_indices:
                    if chunk_idx in web_sources:
                        web_sources[chunk_idx]["contexts"].append(
                            {
                                "text": support.segment.text,
                                "confidence": support.confidence_scores[0],
                            }
                        )

        # Convert to list and filter out sources with no contexts
        sources = [source for source in web_sources.values() if source["contexts"]]

    # Prepare result data
    result_data = {
        "content": text,
        "sources": sources,
        "query": keyword,
    }
    
    if web_search_queries:
        result_data["web_search_query"] = web_search_queries

    return result_data 