from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tool import logger
import uvicorn
from dotenv import load_dotenv
import math
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from document import extract_document_info, analysis_document
from main_chat import cosine_similarity, make_context_from_hits, build_rag_prompt, create_embedding, chat_answer, generate_title_from_messages
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# In-Memory Vector Store (RAG)
# =========================
STORE: List[Dict[str, Any]] = []

# =========================
# Pydantic Models
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

class AskRequest(BaseModel):
    question: str
    context: str | None = None
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    used_chunks: List[str] = []
    
class TitleRequest(BaseModel):
    # Spring이 최근 메시지들을 합쳐서 raw로 넘기면 됨
    # 예: "user: 전세계약 특약 어떻게 써?\nassistant: ...\nuser: ... "
    raw: str

class TitleResponse(BaseModel):
    title: str

class AnalyzeRequest(BaseModel):
    image_path: str



@app.post("/document/analyze")

async def analyze_document(request: Request):
    data = await request.json()
    image_path = data.get("image_path")
    
    print('-> data:', data)
    # 여기에 문서 분석 로직 추가
    result=extract_document_info(image_path)
    parsed = json.loads(result)
    analysis = analysis_document(parsed)
    return analysis

@app.post("/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest):
    vec = create_embedding(req.text)
    return EmbeddingResponse(embedding=vec)

@app.post("/ingest")
def ingest(req: IngestRequest):
    count = 0
    for d in req.docs:
        emb = create_embedding(d.text)
        STORE.append({
            "id": d.id or f"doc_{len(STORE)+1}",
            "text": d.text,
            "meta": d.meta or {},
            "embedding": emb
        })
        count += 1
    return {"inserted": count}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q_emb = create_embedding(req.question)

    scored = [(cosine_similarity(q_emb, d["embedding"]), d) for d in STORE]
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [d for _, d in scored[:req.top_k]]

    rag_context = make_context_from_hits(hits)

    # ✅ Spring에서 넘어온 세션 context + RAG 검색 context 결합
    merged_context = ""
    if req.context:
        merged_context += "=== [세션 참고자료] ===\n" + req.context.strip() + "\n\n"
    merged_context += "=== [RAG 검색자료] ===\n" + (rag_context.strip() if rag_context else "(없음)")

    answer = chat_answer(build_rag_prompt(req.question, merged_context))

    return AskResponse(
        answer=answer,
        used_chunks=[h["text"] for h in hits]
    )
    
@app.post("/title", response_model=TitleResponse)
def make_title(req: TitleRequest):
    raw = (req.raw or "").strip()
    if not raw:
        return TitleResponse(title="새 대화")

    title = generate_title_from_messages(raw)
    if not title:
        title = "새 대화"

    return TitleResponse(title=title)
  


=======
async def analyze_document_endpoint(req: AnalyzeRequest):
    try:
        result = analyze_document(req.image_path)
        return {"analysis": result}
    except Exception as e:
        logger.error("문서 분석 실패", exc_info=e)
        raise HTTPException(status_code=500, detail="문서 분석 실패")
>>>>>>> f39006a63b7fe50d86f48d8570bb1262e6e4e460
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
