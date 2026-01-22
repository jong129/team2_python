# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import time
import base64
import binascii

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
    importanceScore: float
    reason: str

class ChecklistScoreResponse(BaseModel):
    scores: List[ChecklistScoreResult]

# =========================
# Board AI Models (from boardaimain.py)
# =========================
class BoardAiRequest(BaseModel):
    prompt: str = Field(..., description="DB에서 가져온 프롬프트 텍스트")
    title: Optional[str] = Field("", description="게시글 제목")
    content: str = Field(..., description="게시글 본문(또는 사용자가 대충 써둔 초안)")

    truncate: bool = True
    max_chars: int = 8000


class BoardAiImageRequest(BaseModel):
    prompt: str = Field(..., description="DB에서 가져온 프롬프트 텍스트")
    imageBase64: str = Field(..., description="base64 인코딩된 이미지 (dataURL prefix 가능)")
    filename: Optional[str] = Field("", description="원본 파일명")
    contentType: Optional[str] = Field("", description="image/png 등")


class BoardAiResponse(BaseModel):
    resultText: str
    score: Optional[float] = None
    modelName: Optional[str] = None
    tokensIn: Optional[int] = None
    tokensOut: Optional[int] = None
    tokensTotal: Optional[int] = None
    latencyMs: Optional[int] = None

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
# Board AI Helpers (from boardaimain.py)
# =========================
def _cut_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars]
    return s


def _to_board_ai_response(detail: dict) -> BoardAiResponse:
    return BoardAiResponse(
        resultText=detail.get("content", "") or "",
        score=detail.get("score", None),
        modelName=detail.get("model"),
        tokensIn=detail.get("tokens_in"),
        tokensOut=detail.get("tokens_out"),
        tokensTotal=detail.get("tokens_total"),
        latencyMs=detail.get("latency_ms"),
    )

def _call_board_llm_text(full_prompt: str, max_tokens: int = 850) -> dict:
    try:
        t0 = time.perf_counter()

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = os.getenv("CHAT_MODEL", "gpt-4o-mini")

        r = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "너는 정확한 게시판 도우미 AI다. 과장/추측을 금지하고, 요청한 출력 형식을 지킨다."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        content = (r.choices[0].message.content or "").strip()

        usage = getattr(r, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
        tokens_out = getattr(usage, "completion_tokens", None) if usage else None
        tokens_total = getattr(usage, "total_tokens", None) if usage else None
        used_model = getattr(r, "model", None) or model_name

        return {
            "content": content,
            "model": used_model,
            "tokens_in": int(tokens_in) if tokens_in is not None else None,
            "tokens_out": int(tokens_out) if tokens_out is not None else None,
            "tokens_total": int(tokens_total) if tokens_total is not None else None,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")


def _call_board_llm_image(full_prompt: str, data_url: str, max_tokens: int = 350) -> dict:
    try:
        t0 = time.perf_counter()

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = os.getenv("CHAT_MODEL", "gpt-4o-mini")

        r = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "너는 이미지 업로드 정책 판별기다. 반드시 요구한 JSON만 출력한다. 다른 텍스트 금지."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        content = (r.choices[0].message.content or "").strip()

        usage = getattr(r, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
        tokens_out = getattr(usage, "completion_tokens", None) if usage else None
        tokens_total = getattr(usage, "total_tokens", None) if usage else None
        used_model = getattr(r, "model", None) or model_name

        return {
            "content": content,
            "model": used_model,
            "tokens_in": int(tokens_in) if tokens_in is not None else None,
            "tokens_out": int(tokens_out) if tokens_out is not None else None,
            "tokens_total": int(tokens_total) if tokens_total is not None else None,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")


def _decode_base64_image(image_b64: str) -> bytes:
    b64 = (image_b64 or "").strip()
    if b64.startswith("data:") and "base64," in b64:
        b64 = b64.split("base64,", 1)[1].strip()

    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="imageBase64 decode failed")


def _make_data_url(image_bytes: bytes, content_type: str) -> str:
    ct = (content_type or "").strip().lower()
    if not ct.startswith("image/"):
        ct = "image/png"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{ct};base64,{b64}"

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

# =========================
# Board AI Routes (from boardaimain.py)
# =========================
@app.post("/board/summary", response_model=BoardAiResponse)
def board_summary(req: BoardAiRequest):
    title = _cut_text(req.title, 500)
    content = _cut_text(req.content, req.max_chars if req.truncate else 0)

    full_prompt = f"""{req.prompt}

[제목]
{title}

[본문]
{content}

요구사항:
- 한국어로 작성
- 핵심만 3~5줄로 요약
- 과장/추측/투자조언 금지
- 사실만 간결하게
"""
    detail = _call_board_llm_text(full_prompt)
    return _to_board_ai_response(detail)


@app.post("/board/sentiment", response_model=BoardAiResponse)
def board_sentiment(req: BoardAiRequest):
    title = _cut_text(req.title, 500)
    content = _cut_text(req.content, req.max_chars if req.truncate else 0)

    full_prompt = f"""{req.prompt}

[제목]
{title}

[본문]
{content}

요구사항:
- 부동산 시장 관점에서 '호재/악재/혼합' 중 하나로 판단
- 관점은 부동산 직종인이 아닌 집을 구하려는 일반 소비자 관점
- 출력 형식:
  1) 결론: (호재/악재/혼합)
  2) 근거: 2~3줄
- 과장/추측/투자조언 금지
"""
    detail = _call_board_llm_text(full_prompt)
    return _to_board_ai_response(detail)


@app.post("/board/write", response_model=BoardAiResponse)
def board_write(req: BoardAiRequest):
    title = _cut_text(req.title, 500)
    content = _cut_text(req.content, req.max_chars if req.truncate else 0)

    if not title and not content:
        raise HTTPException(status_code=400, detail="title/content 둘 중 하나는 필요합니다.")

    full_prompt = f"""{req.prompt}

[사용자 입력 제목(있으면 참고)]
{title}

[사용자 입력 본문(초안)]
{content}

요구사항:
- 한국어로 작성
- 사용자 입력(제목/본문)에 포함된 사실만 사용 (없는 내용은 만들지 말 것)
- 인터넷 커뮤니티에 쓸법한 자연스러운 문체
- 과장/추측/투자조언/홍보문구 금지
- 게시글 초안 형태로 자연스럽게 문장을 다듬기
- 결과는 '완성된 본문'만 출력
- 길이는 너무 길지 않게 8~20줄 내에서 상황에 맞게
"""
    detail = _call_board_llm_text(full_prompt)
    return _to_board_ai_response(detail)


@app.post("/board/moderate-image", response_model=BoardAiResponse)
def board_moderate_image(req: BoardAiImageRequest):
    image_bytes = _decode_base64_image(req.imageBase64)
    data_url = _make_data_url(image_bytes, req.contentType)

    filename = _cut_text(req.filename or "", 300)
    content_type = _cut_text(req.contentType or "", 100)

    full_prompt = f"""{req.prompt}

[파일명]
{filename}

[Content-Type]
{content_type}

작업:
- 이미지를 보고 게시판 업로드를 허용할지 판단한다.
- 기준 카테고리:
  - AD/COMMERCIAL: 광고/상업 홍보(전단, 가격표, 연락처/URL/QR, 상호/로고 과도 등)
  - SEXUAL: 선정적/노출/성적 암시
  - VIOLENCE: 폭력/잔혹/혐오감 유발
  - HATE: 혐오 표현/상징
  - OTHER: 기타 부적절(사칭/불법/불쾌한 장면 등)
- 애매하면 allowed=false(보수적)로 판단한다.
- score는 판단 확신도(0.0~1.0).
- 한국 영상물 등급 심의 기준 15세 이하의 수준까지는 인정
출력 규칙(절대 준수):
- 반드시 JSON 한 줄만 출력한다. 다른 텍스트 금지.
- 키 이름 고정:
  {{"allowed":true|false,"reason_code":"AD|COMMERCIAL|SEXUAL|VIOLENCE|HATE|OTHER","reason_text":"한글 1줄","score":0.0}}
"""
    detail = _call_board_llm_image(full_prompt, data_url, max_tokens=250)

    if not detail.get("content"):
        raise HTTPException(status_code=502, detail="AI response empty")

    return _to_board_ai_response(detail)

# =========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
