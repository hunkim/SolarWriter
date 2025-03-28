from typing import Dict, List, Any, Optional
import streamlit as st
import json
from langchain_upstage import ChatUpstage 
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime
import asyncio

# Import our new modules
from webcrawler import get_web_content
from llm import llm
from dp import doc_parse
from outline import OUTLINE_TEMPLATES

solarpro = ChatUpstage(model="solar-pro")


def reset_state():
    st.session_state.chat_state = {
        "stage": "init",
        "topic": None,
        "context": None,
        "outline": None,
        "sections": {},
        "refined_article": None,
        "polished_article": None,
        "messages": [],
        "current_article": None,
        "language": None,
    }


def write_section(
    topic: str,
    section_title: str,
    key_points: List[str],
    writing_prompt: str,
    context: str = "",
    example_article: str = "",
):
    # Add style guidance if example article is provided
    style_guidance = ""
    if example_article:
        style_guidance = """
Follow the writing style, tone, and manner of this example article:
{example_article}

Match its:
- Level of formality
- Sentence structure
- Technical depth
- Overall tone
"""

    prompt = f"""For this topic: {topic}. 
    
{f'Using this context as reference:\n{context}\n' if context else ''}
{style_guidance if example_article else ''}
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
            use_google_search=True,  # Enable search for up-to-date information
        )
        
        # Assuming llm function returns a dict with 'content' or 'text' key
        content = response.get("content") or response.get("text")
        
        if not content:
            raise ValueError("No content generated")

        return {
            "success": True,
            "content": content,
            "metadata": {
                "topic": topic,
                "section_title": section_title,
                "key_points": key_points,
                "search_results": response.get(
                    "search_results", []
                ),  # Include if available
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metadata": {
                "topic": topic,
                "section_title": section_title,
                "key_points": key_points,
            },
        }


def polish_full_article(topic: str, article_content: str, example_article: str = "") -> str:
    """Polish the article focusing on section name consistency and text fluency"""
    style_guidance = ""
    if example_article:
        style_guidance = f"""
Reference example for style and tone:
{example_article}

Match its writing style while maintaining the article's unique content."""

    polish_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an expert editor focusing on text fluency and structure consistency. 
Your task is to polish this article while:
1. Ensuring section names are consistent in style and format
2. Improving text fluency and readability
3. Maintaining professional tone
4. Preserving all technical content and accuracy

{style_guidance if example_article else ''}

Focus on making the text flow naturally while keeping the technical depth and original language.""",
            ),
            (
                "user",
                """
                Topic: {topic}

Article content:
{article_content}

Please polish this article focusing on section name consistency and text fluency.
Keep the same language as the input article (do not translate).
Return only the polished article without any additional text.""",
            ),
        ]
    )

    try:
        # Use solar_pro for polishing
        polish_chain = polish_prompt | solarpro | StrOutputParser()
        return polish_chain.stream({"article_content": article_content, "topic": topic})
    except Exception as e:
        st.warning(f"⚠️ Final polishing step skipped: {str(e)}")
        # Return an empty generator for error case
        return (chunk for chunk in [article_content])


def refine_full_article(
    topic: str, 
    article_content: str, 
    topic_content: str = "",
    example_article: str = ""
) -> str:
    style_guidance = ""
    if example_article:
        style_guidance = f"""
Use this example article as a reference for style and tone:
{example_article}

When refining, maintain similar:
- Writing style and voice
- Paragraph structure
- Transition patterns
- Level of detail and explanation
"""

    prompt = f"""As an expert editor, refine this article about '{topic}' to:
1. Improve flow and transitions between sections
2. Ensure consistency in tone and style
3. Strengthen connections between ideas
4. Add cohesive elements
5. Maintain professional language while being engaging
6. Organize structures and rename sections to make it more engaging and easy to read

{style_guidance if example_article else ''}
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
            use_google_search=False,  # No need for search in refinement
        )
        return response.get("content") or response.get("text") or article_content
    except Exception as e:
        st.warning(f"⚠️ Final refinement step skipped: {str(e)}")
        return article_content


def apply_feedback(article_content: str, current_article: str, feedback: str) -> str:
    """Apply user feedback to improve the article"""
    feedback_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert editor who carefully considers user feedback.
Your task is to improve the article based on the specific feedback while:
1. Maintaining the original language and technical accuracy
2. Preserving the overall structure
3. Keeping the professional tone
4. Making only the changes suggested in the feedback

Focus on addressing the feedback points while preserving the article's strengths.""",
            ),
            (
                "user",
                """Original article content:
{article_content}

Current article content:
{current_article}

User feedback:
{feedback}

Please revise the article based on this feedback.
Keep the same language as the input article.
Return only the revised article without any additional text.""",
            ),
        ]
    )

    try:
        feedback_chain = feedback_prompt | solarpro | StrOutputParser()
        return feedback_chain.stream(
            {
                "article_content": article_content,
                "current_article": current_article,
                "feedback": feedback,
            }
        )
    except Exception as e:
        st.warning(f"⚠️ Feedback application skipped: {str(e)}")
        return (chunk for chunk in [article_content])


def refine_outline(topic: str, outline: str):
    # First, enhance the outline
    enhance_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert content strategist. Given a topic and an outline, your task is to make subtle refinements for the given topic to make it more easy to read and enjoyable. 

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
}}""",
            ),
            (
                "user",
                """Topic: {topic}

Original Outline:
{outline}

Please make subtle refinements while keeping the original structure.""",
            ),
        ]
    )

    # Chain for enhancing outline
    enhance_chain = enhance_prompt | solarpro | JsonOutputParser()
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Get enhanced outline
            result = enhance_chain.invoke({"topic": topic, "outline": outline})
            
            # Format the result for better readability
            formatted_result = {
                "topic": topic,
                "original_outline": outline,
                "enhanced_version": result["enhanced_outline"],
                "section_prompts": [],
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
                    "prompt": section_prompt,
                })

            return formatted_result

        except Exception as e:
            if attempt == max_attempts - 1:  # Last attempt failed
                st.warning(f"⚠️ Failed to process outline after {max_attempts} attempts")
                return {
                    "error": f"Failed to process outline: {str(e)}",
                    "topic": topic,
                    "original_outline": outline,
                }
            st.info(f"Attempt {attempt + 1} failed, retrying...")

def detect_language(text: str) -> str:
    """Detect the primary language of the text using SolarPro.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language code (e.g., 'ko', 'en', 'ja', 'zh', 'es', etc.)
    """
    detect_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a language detection expert.
Your task is to determine the primary language of the given text.
Return ONLY the ISO 639-1 language code (e.g., 'ko' for Korean, 'en' for English, 'ja' for Japanese, 'zh' for Chinese, 'es' for Spanish, etc.)"""),
        ("user", """Text to analyze:
{text}

Return ONLY the language code without any other text.""")
    ])

    try:
        detect_chain = detect_prompt | solarpro | StrOutputParser()
        result = detect_chain.invoke({"text": text}).strip().lower()
        
        # Return the detected language code
        return result if result else 'en'  # default to English if empty response
        
    except Exception as e:
        st.warning(f"⚠️ Language detection failed: {str(e)}")
        return 'en'  # default to English on error

async def get_topic_with_context(
    topic: str, url: Optional[str] = None, uploaded_file=None
) -> Dict[str, str]:
    """Get topic and any additional context from URLs/PDFs"""
    context = []
    sources = []
    url_content = ""
    pdf_content = ""

    # Get URL content if provided
    if url:
        st.info(f"🌐 Processing URL: {url}")
        try:
            web_content = await get_web_content(url)
            if web_content:
                # Extract the text content from the tuple if it's a tuple
                url_content = (
                    web_content[0] if isinstance(web_content, tuple) else web_content
                )
                if url_content:
                    context.append(url_content)
                    sources.append(f"URL: {url}")
        except Exception as e:
            st.warning(f"⚠️ Error processing URL: {str(e)}")

    # Get PDF/doc content if provided
    if uploaded_file:
        st.info(f"📄 Processing document: {uploaded_file.name}")
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
        "pdf_content": pdf_content,
    }


def write_article(topic, user_outline, url, uploaded_file, example_article="", example_article_url=""):
    st.subheader("1️⃣ Gathering Context")
    st.spinner("🔄 Getting context...")
    if st.session_state.chat_state["language"] is None:
        language = detect_language(topic)
        st.session_state.chat_state["language"] = language

    # Display detected language with an icon
    language_prompt = f"🌏 Write  in {st.session_state.chat_state['language'].upper()}"
    if st.session_state.chat_state["language"] == "ko":
        language_prompt = "🇰🇷 Write in Korean"    
    elif st.session_state.chat_state["language"] == "en":
        language_prompt = "🌎 Write in English"
    
    st.info(language_prompt)

    # Add language prompt to topic
    topic = topic + " " + language_prompt

    # Check and fill context if needed
    if (
        not st.session_state.chat_state["context"]
        or st.session_state.chat_state["context"].get("context") is None
    ):
        with st.spinner("Processing context..."):
            if example_article_url:
                try:
                    web_content = asyncio.run(get_web_content(example_article_url))
                    st.info(f"🌐 Processing example article URL: {example_article_url}")
                    # Extract text content from web_content, handling tuple case
                    url_article = web_content[0] if isinstance(web_content, tuple) else web_content
                    example_article += url_article if url_article else ""
                except Exception as e:
                    st.warning(f"⚠️ Error processing example article URL: {str(e)}")
                    example_article_url = None
                
            topic_data = asyncio.run(get_topic_with_context(topic, url, uploaded_file))
            st.session_state.chat_state.update({"topic": topic, "context": topic_data})

    if example_article:
        with st.expander("📝 Example Article", expanded=False):
            st.write(example_article)

    # Let's write context
    with st.expander("📄 Context Information", expanded=False):
        st.json(st.session_state.chat_state["context"])

    # Check and fill outline if needed
    st.subheader("2️⃣ Enhancing Outline")
    if not st.session_state.chat_state["outline"] and user_outline:
        with st.spinner("Enhancing outline..."):
            outline_result = refine_outline(
                st.session_state.chat_state["topic"], user_outline
            )
            st.session_state.chat_state["outline"] = outline_result

    assert st.session_state.chat_state["outline"] is not None

    with st.expander("📑 Enhanced Outline", expanded=False):
        st.json(st.session_state.chat_state["outline"]["enhanced_version"])

    # Check and fill sections if needed
    st.subheader("3️⃣ Writing Sections")
    # Create placeholders for all sections first
    section_placeholders = {}
    for section in st.session_state.chat_state["outline"]["enhanced_version"][
        "sections"
    ]:
        with st.expander(f"📄 {section['section_title']}", expanded=True):
            section_placeholders[section["section_title"]] = {
                "status": st.empty(),
                "content": st.empty(),
            }
            section_placeholders[section["section_title"]]["status"].info(
                "⏳ Waiting to generate..."
            )

    # Generate sections one by one
    combined = ""
    for section in st.session_state.chat_state["outline"]["enhanced_version"][
        "sections"
    ]:
        placeholders = section_placeholders[section["section_title"]]
        placeholders["status"].info("🔄 Generating...")

        if (
            st.session_state.chat_state["sections"].get(section["section_title"])
            is None
        ):
            content = write_section(
                topic=st.session_state.chat_state["topic"],
                section_title=section["section_title"],
                key_points=section["key_points"],
                writing_prompt=section["writing_prompt"],
                context=st.session_state.chat_state["context"]["context"],
                example_article=example_article
            )

            st.session_state.chat_state["sections"][section["section_title"]] = {
                "content": content["content"],
                "sources": content["metadata"].get("search_results", []),
            }

        placeholders["status"].success("✅ Generation complete!")
        placeholders["content"].markdown(
            st.session_state.chat_state["sections"][section["section_title"]]["content"]
        )
        combined += f"## {section['section_title']}\n\n{st.session_state.chat_state['sections'][section['section_title']]['content']}\n\n"

    # Continue with refinement immediately after sections are generated
    st.subheader("4️⃣ Refining Article")
    if not st.session_state.chat_state["refined_article"]:
        with st.spinner("🔄 Improving article flow and coherence..."):
            refined = refine_full_article(
                st.session_state.chat_state["topic"],
                combined,
                st.session_state.chat_state["context"]["context"],
                example_article
            )
            st.session_state.chat_state["refined_article"] = refined

    assert st.session_state.chat_state["refined_article"] is not None
    with st.expander("📝 Refined Version", expanded=True):
        st.markdown(st.session_state.chat_state["refined_article"])

    # Continue with polishing immediately after refinement
    st.subheader("5️⃣ Final Polish")
    with st.expander("✨ Polished Version", expanded=True):

        if not st.session_state.chat_state["polished_article"]:
            with st.spinner("✨ Polishing article..."):
                polished = polish_full_article(
                    topic,
                    st.session_state.chat_state["refined_article"],
                    example_article
                )
                polished = st.write_stream(polished)
                st.session_state.chat_state["polished_article"] = polished
                st.session_state.chat_state["current_article"] = polished

        else:
            st.markdown(st.session_state.chat_state["polished_article"])

    # Show feedback interface once article is polished
    if st.session_state.chat_state["polished_article"]:
        st.markdown("---")
        st.subheader("💬 Feedback & Refinement")

        # Show chat history
        for message in st.session_state.chat_state["messages"]:
            with st.chat_message(
                "user" if isinstance(message, HumanMessage) else "assistant"
            ):
                st.markdown(message.content)

        # Handle new feedback
        if feedback := st.chat_input("Provide feedback to improve the article..."):
            # Add user feedback to chat
            st.session_state.chat_state["messages"].append(
                HumanMessage(content=f"💬 **Feedback:** {feedback}")
            )

            with st.chat_message("user"):
                st.markdown(f"💬 **Feedback:** {feedback}")

            with st.chat_message("assistant"):
                with st.spinner("Applying feedback..."):
                    revised = apply_feedback(
                        st.session_state.chat_state["polished_article"],
                        st.session_state.chat_state["current_article"],
                        feedback,
                    )

                    revised = st.write_stream(revised)
                    st.session_state.chat_state["current_article"] = revised
                    st.session_state.chat_state["messages"].append(
                        AIMessage(content=revised)
                    )


def main():
    # Initialize session state
    if "chat_state" not in st.session_state:
        reset_state()

    st.title("🤖 SolarWriter")
    st.markdown(
        "Powered by [Upstage](https://console.upstage.ai)'s advanced LLM and Document AI technology"
    )

    # Single text area for topic and description
    topic = st.text_area(
        "🎯 Topic & Description:",
        value="업스테이지 Document AI 관련 기술 블로그",
        height=100,
        help="Enter your topic and brief description (optional)",
        placeholder="Example:\nGPT-4 Technical Deep Dive\nor\nAI Image Generation - A comprehensive guide for beginners",
    )

    # Context inputs in expandable section
    with st.expander("📚 Additional Context (Optional)", expanded=True):
        url = st.text_input(
            "🌐 Reference URL:",
            value="https://upstage.ai",
            help="Add a URL for reference content",
        )
        uploaded_file = st.file_uploader(
            "📄 Upload Document:", type=["pdf", "docx", "txt"]
        )
        example_article = st.text_area(
            "📝 Example Article (Optional):",
            help="Paste an example article to match its style and tone",
            height=200,
            placeholder="Paste a sample article that demonstrates the writing style you want to follow..."
        )

        example_article_url = st.text_input(
            "📝 Example Article URL (Optional):",
            help="Paste an example article to match its style and tone",
            placeholder="https://example.com/article"
        )

    # Outline selection and customization
    outline_type = st.selectbox("Select content type:", list(OUTLINE_TEMPLATES.keys()))
    user_outline = st.text_area(
        "Customize outline (change if needed):",
        value=OUTLINE_TEMPLATES[outline_type],
        height=300,
    )

    if st.button("🔄 Write Article"):
        reset_state()
        write_article(topic, user_outline, url, uploaded_file, example_article, example_article_url)
    elif st.session_state.chat_state["outline"]:
        write_article(topic, user_outline, url, uploaded_file, example_article, example_article_url)


if __name__ == "__main__":
    main()
    