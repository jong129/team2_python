# boardaimain.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from chatbot import chat_answer_detail

app = FastAPI()

# CORS
# - credentials 포함 요청을 쓸 거면 "*" 대신 실제 Origin을 넣는 게 안전함
# - 네 환경 기준으로 아래 3개 정도는 넣어두는 걸 추천
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
    content: str = Field(..., description="게시글 본문(또는 사용자가 대충 써둔 초안)")

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
    if max_chars and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars]
    return s


def _call_llm(full_prompt: str) -> dict:
    try:
        return chat_answer_detail(full_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")


def _to_response(detail: dict) -> BoardAiResponse:
    return BoardAiResponse(
        resultText=detail.get("content", "") or "",
        score=None,
        modelName=detail.get("model"),
        tokensIn=detail.get("tokens_in"),
        tokensOut=detail.get("tokens_out"),
        tokensTotal=detail.get("tokens_total"),
        latencyMs=detail.get("latency_ms"),
    )


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
    return _to_response(detail)


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

    detail = _call_llm(full_prompt)
    return _to_response(detail)


# =========================
# NEW: WRITE (초안 생성)
# =========================
@app.post("/board/write", response_model=BoardAiResponse)
def board_write(req: BoardAiRequest):
    """
    사용자가 글쓰기에서 대충 적은 title/content를 바탕으로,
    사실 범위 안에서 자연스럽게 "게시글 초안"을 만들어줌.
    """
    title = _cut_text(req.title, 500)
    content = _cut_text(req.content, req.max_chars if req.truncate else 0)

    # 입력이 너무 비면 실패 처리(원하면 완화 가능)
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
- 과장/추측/투자조언/홍보문구 금지
- 게시글 초안 형태로 자연스럽게 문장을 다듬기
- 결과는 '완성된 본문'만 출력 (서론/본론/마무리 정도로 문단 구성)
- 길이는 너무 길지 않게 8~20줄 내에서 상황에 맞게
"""

    detail = _call_llm(full_prompt)
    return _to_response(detail)


if __name__ == "__main__":
    # 지금은 8000으로 실행 중이면 그대로 두면 됨
    # 만약 팀 메인 서버도 8000이면 충돌나니 8002 같은 포트를 쓰는 게 안전
    uvicorn.run("boardaimain:app", host="0.0.0.0", port=8000, reload=True)