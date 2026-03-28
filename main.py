from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uvicorn
import os

# Load from Render secret file path OR local .env
load_dotenv('/etc/secrets/.env')  # Render secret files path
load_dotenv()                      # fallback for local development

# ─── SETUP ──────────────────────────────────────────────
groq_client     = None
pc              = None
embedding_model = None

app = FastAPI(
    title="AI Developer API",
    description="Real AI endpoints — LLM + RAG",
    version="2.0.0"
)

@app.on_event("startup")
async def startup():
    global groq_client, pc, embedding_model
    print("Loading models...")
    groq_client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
    pc              = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Ready!")

# ─── MODELS ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    system:     Optional[str] = "You are a helpful assistant."
    max_tokens: Optional[int] = 500

class ChatResponse(BaseModel):
    response: str
    model:    str
    tokens:   int

class RAGRequest(BaseModel):
    question: str
    top_k:    Optional[int] = 3

class RAGResponse(BaseModel):
    answer:   str
    sources:  List[str]
    scores:   List[float]

class SummarizeRequest(BaseModel):
    text:       str
    num_points: Optional[int] = 5

# ─── HELPERS ────────────────────────────────────────────
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

def retrieve_from_pinecone(question, index_name, top_k=3):
    try:
        index     = pc.Index(index_name)
        embedding = embedding_model.encode(question).tolist()
        results   = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        chunks  = [m.metadata['text'] for m in results.matches]
        scores  = [m.score for m in results.matches]
        headers = [m.metadata.get('header', 'Unknown') for m in results.matches]
        return chunks, scores, headers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

# ─── ROUTES ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message":   "AI Developer API v2.0",
        "endpoints": ["/chat", "/rag", "/summarize", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "models":  "loaded",
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

@app.post("/rag", response_model=RAGResponse)
def rag(request: RAGRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    chunks, scores, headers = retrieve_from_pinecone(
        request.question,
        "pdf-knowledge-base",
        request.top_k
    )
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context found")
    relevant = [(c, s, h) for c, s, h in zip(chunks, scores, headers) if s > 0.3]
    if not relevant:
        return RAGResponse(
            answer="I don't find relevant information in the document.",
            sources=[],
            scores=[]
        )
    chunks  = [r[0] for r in relevant]
    scores  = [r[1] for r in relevant]
    headers = [r[2] for r in relevant]
    context = "\n\n".join([f"[{h}]\n{c}" for c, h in zip(chunks, headers)])
    try:
        answer, _ = call_llm(
            system="""Answer based ONLY on context.
If not in context say 'I don't find this in the document.'
Mention which section your answer comes from.""",
            user=f"Context:\n{context}\n\nQuestion: {request.question}"
        )
        return RAGResponse(answer=answer, sources=headers, scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

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
    uvicorn.run(app, host="0.0.0.0", port=8000)