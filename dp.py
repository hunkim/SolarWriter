import os
import tempfile
from typing import Optional
from langchain_upstage import UpstageDocumentParseLoader

def doc_parse(file) -> Optional[str]:
    """Parse uploaded document and return content string.
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        Parsed content string from the document or None if parsing fails
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            
            # Use UpstageDocumentParseLoader instead
            layzer = UpstageDocumentParseLoader(file_path, split="page")
            docs = layzer.load()
            
            # Combine all pages into a single string
            content = "\n\n".join(doc.page_content for doc in docs)
            return content
            
    except Exception as e:
        print(f"Error parsing document: {str(e)}")
        return None