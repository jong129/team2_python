# main.py
import os
import math
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import uvicorn  # ✅ 추가
from openai import OpenAI

# =========================
# 기본 설정
# =========================
app = FastAPI(title="Team2 RAG FastAPI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

# =========================
# 간단 In-Memory Vector Store
# =========================
STORE: List[Dict[str, Any]] = []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def make_context_from_hits(hits: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    chunks = []
    total = 0
    for h in hits:
        t = h["text"].strip()
        if not t:
            continue
        piece = f"- {t}"
        if total + len(piece) + 1 > max_chars:
            break
        chunks.append(piece)
        total += len(piece) + 1
    return "\n".join(chunks).strip()

# =========================
# Models
# =========================
class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="임베딩할 텍스트")

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class IngestDoc(BaseModel):
    id: Optional[str] = None
    text: str
    meta: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    docs: List[IngestDoc]

class IngestResponse(BaseModel):
    inserted: int

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    used_chunks: List[str] = []

# =========================
# Prompt
# =========================
def build_rag_prompt(question: str, context: str) -> str:
    return f"""
너는 부동산 계약서/등기부등본을 초보자도 이해할 수 있게 설명하는 전문가다.

[중요 규칙]
- 답변에 '샘플', '예시', '컨텍스트', 'context', '프롬프트' 같은 단어를 절대 쓰지 마.
- 불필요한 인삿말(예: "안녕하세요! 무엇을 도와드릴까요?")은 하지 마.
- 아래 자료는 참고용이다. 그대로 복붙하지 말고 질문에 필요한 핵심만 자연스럽게 요약해 설명해라.
- 확실하지 않으면 단정하지 말고, 확인 방법(어떤 항목을 보면 되는지)을 짧게 안내해라.
- 한국어로 답해라.

[참고 자료]
{context if context else "(참고 자료 없음)"}

[사용자 질문]
{question}

[답변]
""".strip()

# =========================
# OpenAI
# =========================
def create_embedding(text: str) -> List[float]:
    emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    r = client.embeddings.create(model=emb_model, input=text)
    return r.data[0].embedding

def chat_answer(prompt: str) -> str:
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    r = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "너는 정확하고 친절한 부동산 문서 분석 AI다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    return (r.choices[0].message.content or "").strip()

# =========================
# Endpoints
# =========================
@app.get("/")  # ✅ 추가: http://localhost:8000
def hello():
    return {"hello": "FastAPI"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest):
    vec = create_embedding(req.text)
    return EmbeddingResponse(embedding=vec)

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    inserted = 0
    for d in req.docs:
        text = (d.text or "").strip()
        if not text:
            continue
        doc_id = d.id or f"doc_{len(STORE)+1}"
        meta = d.meta or {}
        emb = create_embedding(text)
        STORE.append({"id": doc_id, "text": text, "meta": meta, "embedding": emb})
        inserted += 1
    return IngestResponse(inserted=inserted)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return AskResponse(answer="질문을 입력해 주세요.", used_chunks=[])

    q_emb = create_embedding(question)

    hits = []
    if STORE:
        scored = [(cosine_similarity(q_emb, item["embedding"]), item) for item in STORE]
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [it for _, it in scored[: max(1, req.top_k)]]

    context = make_context_from_hits(hits)
    prompt = build_rag_prompt(question, context)
    answer = chat_answer(prompt)

    used_chunks = [h["text"] for h in hits]
    return AskResponse(answer=answer, used_chunks=used_chunks)

@app.post("/chat", response_model=AskResponse)  # ✅ 추가: Spring이 /chat 호출하면 여기로 받음
def chat(req: AskRequest):
    return ask(req)

# =========================
# Run (python main_chat.py)
# =========================
if __name__ == "__main__":
    # python main_chat.py 로 실행 가능하게
    uvicorn.run("main_chat:app", host="0.0.0.0", port=8000, reload=True)
