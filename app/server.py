from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.agent import app_agent, ingest_text
import io
from pypdf import PdfReader
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Local Agentic RAG API")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('app/static/index.html')

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: list[str] = []

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload a text or PDF document to vector store."""
    try:
        content = await file.read()
        filename = file.filename.lower()

        # -------------------------------
        # PDF File with PyPDF
        # -------------------------------
        if filename.endswith(".pdf"):
            text = ""
            try:
                pdf = PdfReader(io.BytesIO(content))
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except Exception as extract_err:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error reading PDF: {extract_err}"
                )

            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="PDF contains no extractable text (might be scanned PDF)."
                )

        # -------------------------------
        # TXT File
        # -------------------------------
        elif filename.endswith(".txt"):
            text = content.decode("utf-8")

        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Upload only .pdf or .txt"
            )

        result = ingest_text(text, source=file.filename)
        return {"message": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_context")
async def delete_context():
    """Clear all ingested documents and reset context."""
    from app.agent import delete_all_context
    try:
        result = delete_all_context()
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Run the agent workflow."""
    try:
        inputs = {"question": request.question}
        result = app_agent.invoke(inputs)
        return {
            "answer": result["answer"],
            "context": result.get("context", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/test")
async def test():
    """Test the agent workflow working or not."""
    try:
        return {"message": "Agent workflow is working."}
    except Exception as e:
        logging.error(f"Error during test: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))