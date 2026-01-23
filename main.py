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

# =========================
# Checklist AI Services
# =========================
from checklist.checklist_rag import ChecklistRagService
from checklist.checklist_scoring import ChecklistScoringService
from checklist.checklist_summary import ChecklistSummaryService
from checklist.checklist_review import ChecklistReviewService, PostChecklistReviewRequest

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
# Checklist AI Service Init
# =========================
checklist_rag_service = ChecklistRagService(
    pdf_path="전세 계약. 두렵지 않아요 전세 사기 예방 A to Z.pdf",
    txt_path="체크리스트_항목.txt"
)

checklist_scoring_service = ChecklistScoringService(checklist_rag_service)
checklist_summary_service = ChecklistSummaryService()
checklist_review_service = ChecklistReviewService(scoring_service=checklist_scoring_service)

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
# Checklist AI Models
# =========================

# ---------- 만족도 요약 ----------
class ChecklistSummaryRequest(BaseModel):
    templateId: int
    comments: List[str]

class ChecklistSummaryResponse(BaseModel):
    positive: List[str]
    negative: List[str]
    suggestions: List[str]


# ---------- AI 미리보기 ----------
class ChecklistAiPreviewRequest(BaseModel):
    baseItems: List[str]
    phase: str

class ChecklistAiPreviewResponse(BaseModel):
    newItems: List[dict]


# ---------- AI 개선 요약 ----------
class ChecklistImproveSummaryRequest(BaseModel):
    templateId: int
    previewItems: List[dict]
    userStats: List[dict]
    satisfaction: dict

class ChecklistImproveSummaryResponse(BaseModel):
    summaries: List[dict]


# ---------- AI 중요도 스코어링 ----------
class ChecklistScoreItem(BaseModel):
    itemId: int
    title: str
    description: str

class ChecklistScoreRequest(BaseModel):
    items: List[ChecklistScoreItem]

class ChecklistScoreResult(BaseModel):
    itemId: int
    title: str
    importanceScore: float
    reason: str

class ChecklistScoreResponse(BaseModel):
    scores: List[ChecklistScoreResult]

import re
import json
from openai import OpenAI
import os

def is_meaningful_comment(comment: str) -> bool:
    """
    의미 없는 코멘트 필터링
    """
    c = comment.strip()

    if len(c) < 5:
        return False

    if re.fullmatch(r"(.)\1{3,}", c):
        return False

    if re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣ]+", c):
        return False

    meaningless = {"test", "asdf", "qwer", "...", "???", "!!!"}
    if c.lower() in meaningless:
        return False

    return True


def call_llm_for_summary(comments: List[str]) -> dict:
    """
    만족도 코멘트 요약
    """
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
다음은 체크리스트 사용자 만족도 코멘트 목록이다.

반드시 JSON으로만 응답하라.
설명, 마크다운, 코드블록은 절대 포함하지 마라.

형식:
{{
  "positive": ["..."],
  "negative": ["..."],
  "suggestions": ["..."]
}}

코멘트:
{chr(10).join(comments)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 체크리스트 UX 분석 전문가다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    try:
        result = json.loads(response.choices[0].message.content.strip())
        if not result.get("suggestions"):
            result["suggestions"] = ["특이 제안 없음."]
        return result
    except Exception:
        return {
            "positive": [],
            "negative": [],
            "suggestions": ["요약 생성 실패"]
        }

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

# =========================
# Checklist AI Routes
# =========================

@app.post("/checklist/summary", response_model=ChecklistSummaryResponse)
def summarize_checklist(req: ChecklistSummaryRequest):
    """
    관리자용 체크리스트 만족도 요약
    """
    filtered = [c for c in req.comments if is_meaningful_comment(c)]

    if not filtered:
        return ChecklistSummaryResponse(
            positive=[],
            negative=[],
            suggestions=["의미 있는 사용자 코멘트가 없습니다."]
        )

    result = call_llm_for_summary(filtered)

    return ChecklistSummaryResponse(
        positive=result.get("positive", []),
        negative=result.get("negative", []),
        suggestions=result.get("suggestions", []),
    )


@app.post("/checklist/ai/preview", response_model=ChecklistAiPreviewResponse)
def checklist_ai_preview(req: ChecklistAiPreviewRequest):
    """
    PDF 기반 AI 체크리스트 개선 미리보기
    """
    result = checklist_rag_service.generate_new_items(
        base_items=req.baseItems,
        phase=req.phase
    )

    return ChecklistAiPreviewResponse(
        newItems=result.get("new_items", [])
    )


@app.post("/checklist/ai/improve/summary", response_model=ChecklistImproveSummaryResponse)
def checklist_ai_improve_summary(req: ChecklistImproveSummaryRequest):
    """
    AI 개선 체크리스트 항목별 이유 설명
    """
    guideline_result = checklist_rag_service.extract_guidelines()
    guidelines = guideline_result.get("guidelines", [])

    summaries = []

    for item in req.previewItems:
        title = item.get("title")

        guideline = next(
            (g for g in guidelines if g.get("title") and g["title"] in title),
            {
                "title": "전세 계약 사기 예방 일반 기준",
                "importance": "MEDIUM",
                "description": "전세 계약 과정에서 반복적으로 문제가 발생하는 주요 위험 요소",
                "source": "PDF 종합 가이드"
            }
        )

        stat = next(
            (s for s in req.userStats if s.get("itemTitle") == title),
            {}
        )

        reason = checklist_rag_service.explain_item_reason(
            guideline=guideline,
            user_stats=stat,
            satisfaction=req.satisfaction,
            preview_item=item
        )

        summaries.append({
            "title": title,
            "reason": reason
        })

    return ChecklistImproveSummaryResponse(summaries=summaries)


@app.post("/checklist/ai/score", response_model=ChecklistScoreResponse)
def checklist_score(req: ChecklistScoreRequest):
    """
    체크리스트 항목 중요도 스코어링
    """
    items = [
        {
            "itemId": i.itemId,
            "title": i.title,
            "description": i.description
        }
        for i in req.items
    ]

    result = checklist_scoring_service.score_items(items)

    return ChecklistScoreResponse(
        scores=result.get("scores", [])
    )

@app.get("/checklists/pre/session/{session_id}/result")
def get_pre_checklist_result(session_id: int):
    # 1️⃣ 미이행 항목 조회
    not_done_items = ...

    # 2️⃣ PDF 기반 AI 스코어링
    score_result = checklist_scoring_service.score_items(not_done_items)

    total_score = sum(s["importanceScore"] for s in score_result["scores"])

    # 3️⃣ 중요도 상위 3개 추출
    top_reasons = (
        sorted(
            score_result["scores"],
            key=lambda x: x["importanceScore"],
            reverse=True
        )[:3]
    )

    # 4️⃣ 사용자 요약 생성 (summary 서비스 사용)
    summary = checklist_summary_service.summarize_pre_result(
        risk_score=total_score,
        reasons=[r["reason"] for r in top_reasons]
    )

    return {
        "riskScore": total_score,
        "summary": summary["summary"],
        "actions": summary["actions"],
        "topReasons": top_reasons,
        "allReasons": score_result["scores"]
    }

@app.post("/checklists/post/review")
def review_post_checklist(req: PostChecklistReviewRequest):
    """
    POST 체크리스트 현재 상태 AI 리뷰
    - Spring 서버가 NOT_DONE 항목을 전달
    """

    # NOT_DONE 항목 dict로 변환
    not_done_items = [
        {
            "itemId": item.itemId,
            "title": item.title,
            "description": item.description
        }
        for item in req.notDoneItems
    ]

    return checklist_review_service.review_post_status(
        not_done_items=not_done_items,
        total=req.total,
        done=req.done
    )





if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
