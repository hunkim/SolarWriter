import os
import tempfile
from typing import Optional
from langchain_upstage import UpstageDocumentParseLoader
import requests
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
            
                print(docs)
                # Combine all pages into a single string
                content = "\n\n".join(doc.page_content for doc in docs)
                return content
            
    except Exception as e:
        print(f"Error parsing document: {str(e)}")
        return None
    
def raw_dp(file) -> Optional[str]:
    """Raw document parsing function using Upstage Document AI API.
    
    Args:
        file: File object from streamlit uploader
        
    Returns:
        Optional[str]: Response JSON as string or None if failed
    """
 
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        print("Error: UPSTAGE_API_KEY not found")
        return None
 
    url = "https://api.upstage.ai/v1/document-ai/document-parse"
    headers = {"Authorization": f"Bearer {api_key}"}
 
    files = {"document": open(file, "rb")}
    data = {"ocr": "force", "model": "document-parse"}
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    files = ["test.pdf", "test2.pdf"]
    for file in files:
        assert os.path.exists(file), "File does not exist"

        # Test raw_dp
        print(raw_dp(file))

        # Test langchain_upstage
        dp = UpstageDocumentParseLoader(file, split="page")
        docs = dp.load()
        print(docs)