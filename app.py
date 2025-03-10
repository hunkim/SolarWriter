import os
import sys
import re
from typing import Dict, List, Any, Optional
import streamlit as st
import json
from langchain_upstage import ChatUpstage 
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import urllib.parse
from tinydb import TinyDB, Query
from datetime import datetime, timedelta
import hashlib
import time
import asyncio

# Import our new modules
from webcrawler import get_web_content
from llm import llm
from dp import doc_parse

solarpro = ChatUpstage(model="solar-pro")


def write_section(topic: str, section_title: str, key_points: List[str], writing_prompt: str, context: str = ""):
    # Modified prompt to include context when available
    prompt = f"""For this topic: {topic}. 
    
{f'Using this context as reference:\n{context}\n' if context else ''}
Use search for this topic and write a section about '{section_title}' that {writing_prompt}:
- Covers these key points: {', '.join(key_points)}
- Uses clear examples and specific details
- Maintains professional tone
- Creates smooth transitions between ideas

If useful, use tables and diagrams to explain the content.

Please actually write a well-structured section (300-500 words). Do not include any other text than the section content."""

    try:
        # Use llm function with the constructed prompt
        response = llm(
            keyword="",
            prompt=prompt,
            use_google_search=True  # Enable search for up-to-date information
        )
        
        # Assuming llm function returns a dict with 'content' or 'text' key
        content = response.get('content') or response.get('text')
        
        if not content:
            raise ValueError("No content generated")

        return {
            "success": True,
            "content": content,
            "metadata": {
                "topic": topic,
                "section_title": section_title,
                "key_points": key_points,
                "search_results": response.get('search_results', [])  # Include if available
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metadata": {
                "topic": topic,
                "section_title": section_title,
                "key_points": key_points
            }
        }


def refine_full_article(topic: str, article_content: str, topic_content: str = "") -> str:
    prompt = f"""As an expert editor, refine this article about '{topic}' to:
1. Improve flow and transitions between sections
2. Ensure consistency in tone and style
3. Strengthen connections between ideas
4. Add cohesive elements
5. Maintain professional language while being engaging
6. Organize structures and rename sections to make it more engaging and easy to read

{f'Using this topic content as reference:\n{topic_content}\n\n' if topic_content else ''}

Original article:
{article_content}

Please provide the refined version while maintaining the same structure and section headers.
Just return the refined article, do not include any other text.
"""

    try:
        response = llm(
            keyword=topic,
            prompt=prompt,
            use_google_search=False  # No need for search in refinement
        )
        return response.get('content') or response.get('text') or article_content
    except Exception as e:
        st.warning(f"⚠️ Final refinement step skipped: {str(e)}")
        return article_content

def write_full_article(topic: str, outline_result: dict, topic_content: str = ""):
    st.subheader("📝 Generated Article")
    
    # Display article title
    st.markdown(f"# {outline_result['enhanced_version']['title']}")
    st.markdown("---")
    
    article_sections = []
    
    # First, generate all sections with visible progress
    st.subheader("1️⃣ Generating Individual Sections:")
    
    for section in outline_result["enhanced_version"]["sections"]:
        with st.expander(f"🔄 Generating: {section['section_title']}", expanded=True):
            progress_placeholder = st.empty()
            progress_placeholder.info("Starting generation...")
            
            section_content = write_section(
                topic=topic,
                section_title=section["section_title"],
                key_points=section["key_points"],
                writing_prompt=section["writing_prompt"],
                context=topic_content
            )
            
            if section_content["success"]:
                progress_placeholder.success("✅ Generation complete!")
                st.markdown(section_content["content"])
                article_sections.append({
                    "title": section["section_title"],
                    "content": section_content["content"],
                    "sources": section_content["metadata"].get("search_results", [])
                })
            else:
                progress_placeholder.error(f"❌ Error: {section_content['error']}")
    
    # Only proceed if all sections were generated successfully
    if len(article_sections) == len(outline_result["enhanced_version"]["sections"]):
        st.markdown("---")
        st.subheader("2️⃣ Refining Complete Article")
        
        with st.spinner("🔄 Polishing the article for better flow and coherence..."):
            # Combine sections into full article
            full_article = f"# {outline_result['enhanced_version']['title']}\n\n"
            for section in article_sections:
                full_article += f"## {section['title']}\n\n{section['content']}\n\n"
            
            # Refine the complete article with topic content
            refined_article = refine_full_article(topic, full_article, topic_content)
            
            # Create tabs for comparing versions
            tab1, tab2, tab3 = st.tabs(["✨ Refined Version", "📝 Original Version", "📊 Statistics"])
            
            with tab1:
                st.markdown(refined_article)
                if st.button("📋 Copy Refined Article"):
                    st.text_area("Copy refined article:", value=refined_article, height=300)
            
            with tab2:
                st.markdown(full_article)
                if st.button("📋 Copy Original Article"):
                    st.text_area("Copy original article:", value=full_article, height=300)
            
            with tab3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sections", len(article_sections))
                with col2:
                    st.metric("Original Words", len(full_article.split()))
                with col3:
                    st.metric("Refined Words", len(refined_article.split()))
        
        # Download options
        st.markdown("---")
        st.subheader("📥 Download Options")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📝 Download Refined Article (MD)",
                data=refined_article,
                file_name="refined_article.md",
                mime="text/markdown"
            )
        
        with col2:
            # JSON with both versions
            json_content = {
                "title": outline_result['enhanced_version']['title'],
                "topic": topic,
                "original_sections": article_sections,
                "refined_article": refined_article,
                "metadata": {
                    "generation_time": str(datetime.now()),
                    "section_count": len(article_sections),
                    "original_word_count": len(full_article.split()),
                    "refined_word_count": len(refined_article.split())
                }
            }
            st.download_button(
                label="🔧 Download Complete Data (JSON)",
                data=json.dumps(json_content, indent=2),
                file_name="article_data.json",
                mime="application/json"
            )
        
        st.success("✅ Article generation and refinement complete!")
    else:
        st.error("❌ Some sections failed to generate. Please try again.")

def refine_outline(topic: str, outline: str):
    # First, enhance the outline
    enhance_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert content strategist. Given a topic and an outline, your task is to make subtle refinements for the given topic to make it more easy to read and enjoyable. 

Rules:
1. Maintain the original flow and main points
2. Make minimal but impactful improvements by:
   - Adding specific angles relevant to the topic
   - Suggesting 2-3 key points per section
   - Creating focused writing prompts

Do NOT:
- Drastically change the outline structure
- Alter the main theme

Output must be valid JSON in this format:
{{
    "enhanced_outline": {{
        "title": "string (keep similar to original)",
        "sections": [
            {{
                "section_title": "string (keep similar to original)",
                "key_points": ["2-3 specific points only"],
                "writing_prompt": "brief, focused prompt"
            }}
        ]
    }}
}}"""),
        ("user", """Topic: {topic}

Original Outline:
{outline}

Please make subtle refinements while keeping the original structure.""")
    ])

    # Chain for enhancing outline
    enhance_chain = enhance_prompt | solarpro | JsonOutputParser()
    
    try:
        # Get enhanced outline
        result = enhance_chain.invoke({"topic": topic, "outline": outline})
        
        # Format the result for better readability
        formatted_result = {
            "topic": topic,
            "original_outline": outline,
            "enhanced_version": result["enhanced_outline"],
            "section_prompts": []
        }

        # Generate focused writing prompts for each section
        for section in result["enhanced_outline"]["sections"]:
            section_prompt = f"""Write about '{section['section_title']}' focusing on:
- Key points to cover: {', '.join(section['key_points'])}
- Specific examples related to {topic}
- Professional tone
- Clear connection to the main topic

Additional guidance: {section['writing_prompt']}"""
            
            formatted_result["section_prompts"].append({
                "section_title": section["section_title"],
                "prompt": section_prompt
            })

        return formatted_result

    except Exception as e:
        return {
            "error": f"Failed to process outline: {str(e)}",
            "topic": topic,
            "original_outline": outline
        }

async def get_topic_with_context(topic: str, url: Optional[str] = None, uploaded_file = None) -> Dict[str, str]:
    """Get topic and any additional context from URLs/PDFs"""
    context = []
    sources = []
    url_content = ""
    pdf_content = ""
    
    # Get URL content if provided
    if url:
        try:
            web_content = await get_web_content(url)
            if web_content:
                # Extract the text content from the tuple if it's a tuple
                url_content = web_content[0] if isinstance(web_content, tuple) else web_content
                if url_content:
                    context.append(url_content)
                    sources.append(f"URL: {url}")
        except Exception as e:
            st.warning(f"⚠️ Error processing URL: {str(e)}")
    
    # Get PDF/doc content if provided
    if uploaded_file:
        try:
            pdf_content = doc_parse(uploaded_file)
            if pdf_content:
                context.append(pdf_content)
                sources.append(f"File: {uploaded_file.name}")
        except Exception as e:
            st.warning(f"⚠️ Error processing document: {str(e)}")
    
    return {
        "topic": topic,
        "context": "\n\n".join(context),
        "source": ", ".join(sources) if sources else "direct input",
        "url_content": url_content,
        "pdf_content": pdf_content
    }

def main():
    st.title("🤖 AI Article Generator")
    st.markdown("---")
    
    # Get topic
    topic = st.text_input(
        "🎯 Enter your topic:",
        value="업스테이지 Document AI 관련 기술 블로그 (한국어)",
        help="Be specific to get better results"
    )
    
    # Optional additional context
    url = st.text_input(
        "🌐 Optional: Enter URL for additional context",
        value="https://upstage.ai",
        help="Enter a URL to include its content"
    )
    
    uploaded_file = st.file_uploader(
        "📄 Optional: Upload document for additional context",
        type=["pdf", "docx", "txt"]
    )

    # Show extracted content if available
    if url or uploaded_file:
        topic_data = asyncio.run(get_topic_with_context(topic, url, uploaded_file))
        
        # Create columns for URL and PDF content
        col1, col2 = st.columns(2)
        
        with col1:
            if url and topic_data["url_content"]:
                with st.expander("🌐 URL Content", expanded=False):
                    st.markdown("### Content from URL:")
                    st.text_area(
                        "URL content:",
                        value=topic_data["url_content"],
                        height=300,
                        disabled=True
                    )
                    st.caption(f"Source: {url}")
        
        with col2:
            if uploaded_file and topic_data["pdf_content"]:
                with st.expander("📄 Document Content", expanded=False):
                    st.markdown("### Content from Document:")
                    st.text_area(
                        "Document content:",
                        value=topic_data["pdf_content"],
                        height=300,
                        disabled=True
                    )
                    st.caption(f"Source: {uploaded_file.name}")

    use_search = st.checkbox("🔍 Use Google Search", value=True,
                           help="Enable to include recent information")
    
    sample_outline = """Title: 

1. Introduction and Background
   - Problem statement
   - Why this matters

2. Technical Deep Dive
   - Core architecture
   - Key components
   - How it works

3. Implementation Details
   - Setup and configuration
   - Code examples
   - Best practices

4. Performance and Results
   - Benchmarks
   - Real-world examples
   - Lessons learned

5. Future Developments
   - Planned improvements
   - Open challenges
   - Community feedback"""

    user_outline = st.text_area(
        "📝 Enter your article outline:",
        value=sample_outline,
        height=300,
        help="Enter your outline with a title and numbered sections"
    )

    if st.button("🚀 Generate Article") and topic:
        with st.spinner("🔄 Processing input and outline..."):
            # Get topic with context again for article generation
            topic_data = asyncio.run(get_topic_with_context(topic, url, uploaded_file))
            
            # First, refine the outline
            outline_result = refine_outline(topic_data["topic"], user_outline)
            
            if "error" in outline_result:
                st.error(f"❌ Error: {outline_result['error']}")
            else:
                st.success(f"✅ Using context from: {topic_data['source']}")
                with st.expander("🔍 View Refined Outline"):
                    st.json(outline_result["enhanced_version"])
                
                # Pass context to write_full_article
                write_full_article(
                    topic=topic_data["topic"],
                    outline_result=outline_result,
                    topic_content=topic_data["context"]
                )
    elif not topic:
        st.warning("Please enter a topic to continue")

if __name__ == "__main__":
    main()


    