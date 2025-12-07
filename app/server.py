from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.agent import app_agent, ingest_text

app = FastAPI(title="Local Agentic RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: list[str] = []

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload a text file to vector store."""
    try:
        content = await file.read()
        text = content.decode("utf-8")
        result = ingest_text(text, source=file.filename)
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