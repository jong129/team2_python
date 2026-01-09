# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
import os
import tempfile

from document import extract_document_info, analysis_document

# ✅ main_chat에서 기존 함수 import 유지 + chroma 기능 추가
from rag_store import (
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 운영에서는 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    # ✅ 확장(옵션): 안 보내면 기본값 사용
    chunk: Optional[bool] = True
    chunk_size: Optional[int] = 900
    overlap: Optional[int] = 120

class IngestRequest(BaseModel):
    docs: List[IngestDoc]

class AskRequest(BaseModel):
    question: str
    context: str | None = None
    top_k: int = 5

    # ✅ 확장(옵션): 체크리스트/문서 타입 필터
    doc_type: Optional[str] = None   # 예: "checklist", "registry", "contract"
    stage: Optional[str] = None      # 예: "pre", "post"

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

# =========================
# Routes
# =========================
@app.get("/")
def hello():
    return {"hello": "FastAPI", "store": "chroma"}

@app.post("/document/analyze")
async def analyze_document(request: Request):
    """
    ✅ 두 가지 입력을 모두 지원:
    1) JSON: { "image_path": "C:/.../a.jpg" }  (현재 네 방식)
    2) multipart/form-data: file 업로드            (실서비스용)
    """
    content_type = (request.headers.get("content-type") or "").lower()

    # (1) multipart 업로드
    if "multipart/form-data" in content_type:
        form = await request.form()
        up = form.get("file")
        if up is None:
            return {"error": "multipart 요청에는 file 필드가 필요합니다."}

        # UploadFile 대응
        filename = getattr(up, "filename", "") or "upload"
        suffix = os.path.splitext(filename)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await up.read()
            tmp.write(data)
            tmp_path = tmp.name

        try:
            result = extract_document_info(tmp_path)
            parsed = json.loads(result)
            analysis = analysis_document(parsed)
            return analysis
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # (2) JSON image_path 방식
    data = await request.json()
    image_path = data.get("image_path")
    if not image_path:
        return {"error": "JSON 요청에는 image_path가 필요합니다."}

    result = extract_document_info(image_path)
    parsed = json.loads(result)
    analysis = analysis_document(parsed)
    return analysis

@app.post("/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest):
    vec = create_embedding(req.text)
    return EmbeddingResponse(embedding=vec)

@app.post("/ingest")
def ingest(req: IngestRequest):
    """
    ✅ Chroma에 영속 저장
    - 문서/체크리스트/서류 텍스트를 넣어두면 /ask에서 검색됨
    - meta로 doc_type, stage 같은 태그를 넣어두면 필터 검색 가능
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

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q_type = classify_question(req.question)

    # ✅ 간단 질문이면 RAG도 안 탐
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


    # ✅ 분석 질문만 RAG 수행
    q_emb = create_embedding(req.question)

    hits = chroma_search(
        query_embedding=q_emb,
        top_k=req.top_k,
        doc_type=req.doc_type,
        stage=req.stage
    )

    rag_context = make_context_from_hits(hits)

    merged_context = ""
    if req.context:
        merged_context += "=== [세션 참고자료] ===\n" + req.context.strip() + "\n\n"
    merged_context += "=== [RAG 검색자료] ===\n" + (rag_context.strip() if rag_context else "(없음)")

    detail = chat_answer_detail(build_rag_prompt(req.question, merged_context))
    answer = detail["content"]

    # ✅ references 만들기 (chunkId = chroma id)
    references = [
        RagReference(
            chunkId=h["id"],
            title=(h.get("meta") or {}).get("title"),
            snippet=(h.get("text") or "")[:200],
            score=float(h.get("score", 0.0)),
            rankNo=i + 1
        )
        for i, h in enumerate(hits)
    ]

    # ✅ followups 생성 (근거 기반)
    followups = generate_followups(req.question, answer, hits)

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



@app.post("/title", response_model=TitleResponse)
def make_title(req: TitleRequest):
    raw = (req.raw or "").strip()
    if not raw:
        return TitleResponse(title="새 대화")

    title = generate_title_from_messages(raw)
    if not title:
        title = "새 대화"
    return TitleResponse(title=title)

if __name__ == "__main__":
    uvicorn.run("main_chroma:app", host="0.0.0.0", reload=True)
