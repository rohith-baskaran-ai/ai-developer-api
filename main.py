from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from groq import Groq
from pinecone import Pinecone
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

groq_client = None
pc          = None

app = FastAPI(
    title="AI Developer API",
    description="Real AI endpoints — LLM + RAG",
    version="2.0.0"
)

@app.on_event("startup")
async def startup():
    global groq_client, pc
    print("Loading models...")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    pc          = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    print("Ready!")

class ChatRequest(BaseModel):
    message:    str
    system:     Optional[str] = "You are a helpful assistant."
    max_tokens: Optional[int] = 500

class ChatResponse(BaseModel):
    response: str
    model:    str
    tokens:   int

class SummarizeRequest(BaseModel):
    text:       str
    num_points: Optional[int] = 5

def call_llm(system, user, max_tokens=500):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content, response.usage.total_tokens

@app.get("/")
def root():
    return {
        "message":   "AI Developer API v2.0",
        "endpoints": ["/chat", "/summarize", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "version": "2.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(request.message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long")
    try:
        response, tokens = call_llm(
            system=request.system,
            user=request.message,
            max_tokens=request.max_tokens
        )
        return ChatResponse(
            response=response,
            model="llama-3.3-70b-versatile",
            tokens=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

@app.post("/summarize")
def summarize(request: SummarizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(request.text) < 50:
        raise HTTPException(status_code=400, detail="Text too short")
    try:
        summary, tokens = call_llm(
            system=f"Summarize into {request.num_points} clear bullet points.",
            user=request.text[:4000]
        )
        return {
            "summary":         summary,
            "original_length": len(request.text),
            "num_points":      request.num_points,
            "tokens_used":     tokens
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))