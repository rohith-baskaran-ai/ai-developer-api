import requests

base = "http://localhost:8000"

# Test 1 — health
r = requests.get(f"{base}/health")
print("GET /health:", r.json())

# Test 2 — chat
r = requests.post(f"{base}/chat", json={
    "message": "What is RAG in AI? Explain in 2 lines.",
    "max_tokens": 200
})
print("POST /chat:", r.json())

# Test 3 — RAG
r = requests.post(f"{base}/rag", json={
    "question": "What is machine learning?",
    "top_k": 3
})
print("POST /rag:", r.json())

# Test 4 — summarize
r = requests.post(f"{base}/summarize", json={
    "text": """Machine learning is a subset of artificial intelligence
    that enables systems to learn from data. Instead of being explicitly
    programmed, ML models improve through experience. There are three main
    types: supervised, unsupervised, and reinforcement learning.""",
    "num_points": 3
})
print("POST /summarize:", r.json())