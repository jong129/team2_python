# main_checklist.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
import uvicorn
import json
import os
import re

# =========================
# OpenAI Client (v1.x)
# =========================
from openai import OpenAI

# =========================
# RAG Service
# =========================
from checklist.checklist_rag import ChecklistRagService

# =========================
# Scoring Service
# =========================
from checklist.checklist_scoring import ChecklistScoringService

# =========================
# Env
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# App
# =========================
app = FastAPI(title="Checklist AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# RAG Service Init
# =========================
rag_service = ChecklistRagService(
    pdf_path="전세 계약. 두렵지 않아요 전세 사기 예방 A to Z.pdf",
    txt_path="체크리스트_항목.txt"
)

# =========================
# Scoring Service Init
# =========================
scoring_service = ChecklistScoringService(rag_service)

# ==================================================
# Pydantic Models
# ==================================================

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
    previewItems: List[dict]     # AI 미리보기 결과
    userStats: List[dict]        # 항목별 완료/미완료/해당없음
    satisfaction: dict           # 만족도 요약


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

# ==================================================
# Util Functions
# ==================================================
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

    content = response.choices[0].message.content.strip()

    try:
        result = json.loads(content)

        if not result.get("suggestions"):
            result["suggestions"] = ["특이 제안 없음."]

        return result

    except Exception:
        return {
            "positive": [],
            "negative": [],
            "suggestions": ["요약 생성 실패"]
        }


# ==================================================
# Routes
# ==================================================
@app.get("/")
def health():
    return {"service": "checklist-ai", "status": "ok"}


# ---------- 1️⃣ 만족도 요약 ----------
@app.post(
    "/checklist/summary",
    response_model=ChecklistSummaryResponse
)
def summarize(req: ChecklistSummaryRequest):
    """
    관리자용 체크리스트 만족도 요약
    """

    filtered_comments = [
        c for c in req.comments
        if is_meaningful_comment(c)
    ]

    if not filtered_comments:
        return ChecklistSummaryResponse(
            positive=[],
            negative=[],
            suggestions=["의미 있는 사용자 코멘트가 없습니다."]
        )

    result = call_llm_for_summary(filtered_comments)

    return ChecklistSummaryResponse(
        positive=result.get("positive", []),
        negative=result.get("negative", []),
        suggestions=result.get("suggestions", []),
    )


# ---------- 2️⃣ AI 개선 미리보기 ----------
@app.post(
    "/checklist/ai/preview",
    response_model=ChecklistAiPreviewResponse
)
def ai_preview(req: ChecklistAiPreviewRequest):
    """
    PDF 기반 AI 체크리스트 개선 미리보기
    """

    result = rag_service.generate_new_items(
        base_items=req.baseItems,
        phase=req.phase
    )

    return ChecklistAiPreviewResponse(
        newItems=result.get("new_items", [])
    )


# ---------- 3️⃣ AI 개선 요약 (왜 이렇게 개선했는지) ----------
@app.post(
    "/checklist/ai/improve/summary",
    response_model=ChecklistImproveSummaryResponse
)
def ai_improve_summary(req: ChecklistImproveSummaryRequest):
    """
    AI 개선 체크리스트 항목별 '이유 설명'
    """

    guideline_result = rag_service.extract_guidelines()
    guidelines = guideline_result.get("guidelines", [])

    summaries = []

    for preview_item in req.previewItems:
        title = preview_item.get("title")

        # 1️⃣ 가이드라인 매칭 (느슨하게)
        guideline = next(
            (g for g in guidelines
             if g.get("title") and g["title"] in title),
            None
        )

        # ✅ 매칭 실패 시 기본 가이드라인 사용
        if not guideline:
            guideline = {
                "title": "전세 계약 사기 예방 일반 기준",
                "importance": "MEDIUM",
                "description": "전세 계약 과정에서 반복적으로 문제가 발생하는 주요 위험 요소에 대한 예방 기준",
                "source": "PDF 종합 가이드"
            }

        # 2️⃣ 사용자 통계 매칭
        stat = next(
            (s for s in req.userStats if s.get("itemTitle") == title),
            {}
        )

        # 3️⃣ LLM 설명 생성
        reason = rag_service.explain_item_reason(
            guideline=guideline,
            user_stats=stat,
            satisfaction=req.satisfaction,
            preview_item=preview_item
        )

        summaries.append({
            "title": title,
            "reason": reason
        })

    return ChecklistImproveSummaryResponse(summaries=summaries)

# ---------- 4️⃣ 체크리스트 항목 중요도 스코어링 ----------
@app.post(
    "/checklist/ai/score",
    response_model=ChecklistScoreResponse
)
def score_checklist_items(req: ChecklistScoreRequest):
    """
    PDF 기반 전세 사기 예방 가이드를 기준으로
    체크리스트 항목별 중요도 점수를 산출한다.
    """

    # Dict 변환 (ChecklistScoringService 입력 형식)
    items = [
        {
            "itemId": item.itemId,
            "title": item.title,
            "description": item.description
        }
        for item in req.items
    ]

    result = scoring_service.score_items(items)

    return ChecklistScoreResponse(
        scores=result.get("scores", [])
    )


# ==================================================
# Run
# ==================================================
if __name__ == "__main__":
    uvicorn.run(
        "main_checklist:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
