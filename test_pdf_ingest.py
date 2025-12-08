
import io
import os
import sys

# Set environment variables for local testing BEFORE importing app modules
os.environ["WEAVIATE_URL"] = "http://localhost:8080"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

import pypdf
from app.server import ingest_document, app
from fastapi.testclient import TestClient

client = TestClient(app)

def create_dummy_pdf(text_content="Hello, this is a test PDF document."):
    """Creates a dummy PDF file in memory."""
    pdf_content = (
        b"%PDF-1.1\n"
        b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n"
        b"2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n"
        b"3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]/Resources<<>>>>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000117 00000 n\n"
        b"trailer\n<</Size 4/Root 1 0 R>>\nstartxref\n206\n%%EOF"
    )
    return io.BytesIO(pdf_content)

def test_pdf_upload():
    print("Testing PDF upload...", end=" ")
    try:
        pdf_path = "meditation.pdf"
        if os.path.exists(pdf_path):
            print(f"Uploading {pdf_path}...")
            with open(pdf_path, "rb") as f:
                content = f.read()
                filename = pdf_path
        else:
            print(f"Warning: {pdf_path} not found. Creating dummy PDF.")
            content = create_dummy_pdf().getvalue()
            filename = "test.pdf"
        
        response = client.post(
            "/ingest",
            files={"file": (filename, content, "application/pdf")}
        )
        
        if response.status_code == 200:
            print("SUCCESS. Document processed.")
            print(f"Response: {response.json()}")
        else:
            print(f"FAILED. Status: {response.status_code}, Detail: {response.text}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_pdf_upload()
