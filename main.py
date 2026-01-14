# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from typing import Optional

from tool import logger
from document.document import analyze_document, analyze_document_b64
from chatbot import (
    make_context_from_hits,
    build_rag_prompt,
    create_embedding,
    chat_answer,
    chat_answer_detail,
    generate_title_from_messages,
    chroma_add_docs,
    chroma_search,
    classify_question,
    build_simple_prompt,
    generate_followups,
)

load_dotenv()

app = FastAPI()

# CORS 설정 : 프론트/스프링에서 호출 가능하게
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic Models : 요청/응답의 스키마
# =========================
class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="임베딩할 텍스트")

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class IngestDoc(BaseModel):
    id: Optional[str] = None
    text: str
    meta: Optional[Dict[str, Any]] = None

    # 안 보내면 기본값 사용
    chunk: Optional[bool] = True
    chunk_size: Optional[int] = 900
    overlap: Optional[int] = 120

class IngestRequest(BaseModel):
    docs: List[IngestDoc]

class AskRequest(BaseModel):
    question: str
    context: str | None = None  
    top_k: int = 5

    # 체크리스트/문서 타입 필터
    doc_type: Optional[str] = None   # 예: "checklist", "registry", "contract"
    stage: Optional[str] = None      # 예: "pre", "post"
    
    user_id: Optional[str] = None
    doc_id: Optional[str] = None

class RagReference(BaseModel):
    chunkId: str
    title: Optional[str] = None
    snippet: str
    
    score: Optional[float] = None
    rankNo: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    references: List[RagReference] = []
    followUpQuestions: List[str] = []
    
    model: Optional[str] = None
    tokensIn: Optional[int] = None
    tokensOut: Optional[int] = None
    tokensTotal: Optional[int] = None
    latencyMs: Optional[int] = None

class TitleRequest(BaseModel):
    raw: str

class TitleResponse(BaseModel):
    title: str
    
class AnalyzeRequest(BaseModel):
    image_path: Optional[str] = None
    image_b64: Optional[str] = None

# =========================
# Routes
# =========================
@app.get("/")
def hello():
    return {"hello": "FastAPI", "store": "chroma"}

# 문서 이미지 분석 : 이미지 경로 또는 base64를 받아서 텍스트/정보를 추출(등기부등본, 계약서 등)
@app.post("/document/analyze")
async def analyze_document_endpoint(req: AnalyzeRequest):
    try:
        if req.image_b64:
            return analyze_document_b64(req.image_b64)
        if req.image_path:    
            return analyze_document(req.image_path)
    except Exception as e:
        logger.error("문서 분석 실패", exc_info=e)
        raise HTTPException(status_code=500, detail="문서 분석 실패")

# 임베딩 생성 : 텍스트 -> embedding vector(List[float])
@app.post("/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest):
    vec = create_embedding(req.text)
    return EmbeddingResponse(embedding=vec)

# Chroma에 문서 저장(ingest) : 문서/체크리스트/서류 텍스트를 Chroma에 chunk 단위로 저장
# meta 필터(doc_type, stage, user_id, doc_id)로 나중에 검색 조건을 걸 수 있음.
@app.post("/ingest")
def ingest(req: IngestRequest):
    """
    1. 요청의 docs를 순회하면서
    2. 각 doc을 {"id":..., "text":..., "meta":...} 형태로 만들고
    3. chroma_add_docs()에 전달
    4. chunk 옵션은 doc별로 override
    5. insert된 chunk 수를 합산해서 반환
    """
    inserted_total = 0

    for d in req.docs:
        docs = [{
            "id": d.id,
            "text": d.text,
            "meta": d.meta or {}
        }]

        inserted = chroma_add_docs(
            docs=docs,
            chunk=bool(d.chunk) if d.chunk is not None else True,
            chunk_size=int(d.chunk_size) if d.chunk_size else 900,
            overlap=int(d.overlap) if d.overlap else 120,
        )
        inserted_total += inserted

    return {"inserted_chunks": inserted_total}

# RAG 질문응답 : 질문을 분류(simple vs 분석질문) 
# 분석질문이면 Chroma 검색 -> 근거 만들기 -> RAG 프롬프트로 LLM 호출 -> 답변+근거+후속질문+사용량 반환 
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # 1. 질문 분류
    q_type = classify_question(req.question)

    # 2-1. simple 질문 처리(RAG X. 빠르고 저렴)
    if q_type == "simple":
        prompt = build_simple_prompt(req.question)
        answer = chat_answer(prompt)
        detail = chat_answer_detail(prompt)

        # simple은 refs 없으니 fallback으로만 3개 제공(또는 generate_followups 호출 안 해도 됨)
        return AskResponse(
            answer=detail["content"],
            references=[],
            followUpQuestions=[
                "이 내용을 3줄로 더 쉽게 설명해줘",
                "이걸 하려면 준비물/서류가 뭐가 필요해?",
                "내 상황에서 주의할 점 3가지만 알려줘"
            ],
            model=detail["model"],
            tokensIn=detail["tokens_in"],
            tokensOut=detail["tokens_out"],
            tokensTotal=detail["tokens_total"],
            latencyMs=detail["latency_ms"],
        )


    # 2-2. 분석 질문이면 RAG 수행
    # 질문 임베딩
    q_emb = create_embedding(req.question)

    # Chroma 검색
    hits = chroma_search(
        query_embedding=q_emb,
        top_k=req.top_k,
        doc_type=req.doc_type,
        stage=req.stage,
        user_id=req.user_id,
        doc_id=req.doc_id,
    )

    # RAG 컨텍스트 만들기 : hits의 텍스트/메타를 LLM 프롬프트에 넣기 좋은 형태로 변환
    rag_context = make_context_from_hits(hits)
    
    # 세션 컨텍스트 + RAG 검색 컨텍스트 합치기
    merged_context = ""
    if req.context:
        merged_context += "=== [세션 참고자료] ===\n" + req.context.strip() + "\n\n"
    merged_context += "=== [RAG 검색자료] ===\n" + (rag_context.strip() if rag_context else "(없음)")

    # LLM 호출 (상세 메타 포함)
    detail = chat_answer_detail(build_rag_prompt(req.question, merged_context))
    answer = detail["content"]

    # references 만들기 
    references = [
        RagReference(
            chunkId=h["id"],    # chunkId = chroma id(문자열)
            title=(h.get("meta") or {}).get("title"),
            snippet=(h.get("text") or "")[:200],        # 200만 자름(프론트에 길게 안 보내려는 의도)
            score=float(h.get("score", 0.0)),
            rankNo=i + 1    # 검색 순위 (1부터)
        )
        for i, h in enumerate(hits)
    ]

    # followups 생성 (근거 기반)
    followups = generate_followups(req.question, answer, hits)

    # AskResponse로 반환 (usage 포함)
    return AskResponse(
        answer=answer,
        references=references,
        followUpQuestions=followups,
        model=detail["model"],
        tokensIn=detail["tokens_in"],
        tokensOut=detail["tokens_out"],
        tokensTotal=detail["tokens_total"],
        latencyMs=detail["latency_ms"],
    )

# 세션 제목 생성
@app.post("/title", response_model=TitleResponse)
def make_title(req: TitleRequest):
    # raw가 비어있으면 새 대화
    raw = (req.raw or "").strip()
    if not raw:
        return TitleResponse(title="새 대화")

    # 아니면 generate_title_from_messages(raw) 호출
    title = generate_title_from_messages(raw)
    
    # 결과가 비면 fallback "새 대화"
    if not title:
        title = "새 대화"
    return TitleResponse(title=title)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
