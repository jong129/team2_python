# boardaimain.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# 팀 공용 LLM 호출 함수 재사용 (너 main.py에서 쓰던 그대로)
from chatbot import chat_answer_detail

app = FastAPI()

# CORS (main.py 스타일 최대한 맞춤)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic Models
# =========================
class BoardAiRequest(BaseModel):
    prompt: str = Field(..., description="DB에서 가져온 프롬프트 텍스트")
    title: Optional[str] = Field("", description="게시글 제목")
    content: str = Field(..., description="게시글 본문")

    # 옵션: 너무 길면 잘라서 보낼지 (토큰/비용/지연 방지)
    truncate: bool = True
    max_chars: int = 8000

class BoardAiResponse(BaseModel):
    resultText: str
    score: Optional[float] = None
    modelName: Optional[str] = None
    tokensIn: Optional[int] = None
    tokensOut: Optional[int] = None
    tokensTotal: Optional[int] = None
    latencyMs: Optional[int] = None

# =========================
# Helper
# =========================
def _cut_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars and len(s) > max_chars:
        return s[:max_chars]
    return s

def _call_llm(full_prompt: str) -> dict:
    """
    chat_answer_detail()이 반환하는 dict 구조를 그대로 사용
    (main.py에서 detail["content"], detail["model"], detail["tokens_in"] 등 쓰는 패턴)
    """
    try:
        return chat_answer_detail(full_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")

# =========================
# Routes
# =========================
@app.get("/")
def hello():
    return {"hello": "Board AI FastAPI"}

@app.post("/board/summary", response_model=BoardAiResponse)
def board_summary(req: BoardAiRequest):
    title = _cut_text(req.title, 500)
    content = _cut_text(req.content, req.max_chars if req.truncate else 0)

    # 프롬프트 + 입력 통합 (요약 전용)
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

    detail = _call_llm(full_prompt)

    return BoardAiResponse(
        resultText=detail.get("content", ""),
        score=None,
        modelName=detail.get("model"),
        tokensIn=detail.get("tokens_in"),
        tokensOut=detail.get("tokens_out"),
        tokensTotal=detail.get("tokens_total"),
        latencyMs=detail.get("latency_ms"),
    )

@app.post("/board/sentiment", response_model=BoardAiResponse)
def board_sentiment(req: BoardAiRequest):
    title = _cut_text(req.title, 500)
    content = _cut_text(req.content, req.max_chars if req.truncate else 0)

    # 프롬프트 + 입력 통합 (호재/악재 판단 전용)
    full_prompt = f"""{req.prompt}

[제목]
{title}

[본문]
{content}

요구사항:
- 부동산 시장 관점에서 '호재/악재/혼합' 중 하나로 판단
- 출력 형식:
  1) 결론: (호재/악재/혼합)
  2) 근거: 2~3줄
- 과장/추측/투자조언 금지
"""

    detail = _call_llm(full_prompt)

    return BoardAiResponse(
        resultText=detail.get("content", ""),
        score=None,  # 파이썬에서 점수까지 만들고 싶으면 나중에 파싱해서 넣으면 됨
        modelName=detail.get("model"),
        tokensIn=detail.get("tokens_in"),
        tokensOut=detail.get("tokens_out"),
        tokensTotal=detail.get("tokens_total"),
        latencyMs=detail.get("latency_ms"),
    )

if __name__ == "__main__":
    # 팀 main.py(8000)랑 포트 충돌 방지: 8002 추천
    uvicorn.run("boardaimain:app", host="0.0.0.0", port=8000, reload=True)
