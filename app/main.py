from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.utils import get_rag_response

app = FastAPI(
    title="Portfolio Chatbot API",
    description="RAG-based chatbot for Mrigank Singh's portfolio",
    version="1.0.0"
)

# CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mrigank.is-a.dev", "https://mrigank.is-a.dev/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def health_check():
    """Health check endpoint for Render."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - processes user message through RAG pipeline.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response = get_rag_response(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
