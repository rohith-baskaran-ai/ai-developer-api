# AI Developer API

A production-ready AI API built with FastAPI, Docker, and deployed on Render.

## Live API

Base URL: https://ai-developer-api-z6rq.onrender.com

Endpoints:
- GET  /health    → service status + uptime
- GET  /stats     → request count + uptime
- POST /chat      → LLM chat via Groq + Llama3
- POST /summarize → text summarization

## Quick Test
```python
import requests

base = "https://ai-developer-api-z6rq.onrender.com"

# Chat
r = requests.post(f"{base}/chat", json={
    "message": "What is RAG in AI?"
})
print(r.json())
```

## Stack

Python 3.11 · FastAPI · Docker · Render · Groq · Llama3-70b

## What I learned

Deploying AI APIs is harder than deploying regular APIs — memory
limits on free tiers mean you can't load large models like
sentence-transformers. The solution is to use API-based services
(Groq) instead of loading models locally.