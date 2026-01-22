# boardaimain.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import time
import base64
import binascii
import uvicorn

from openai import OpenAI

# =========================
# OpenAI Client
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다. (Windows 사용자 변수/시스템 변수 설정 후 터미널 재시작)")

client = OpenAI(api_key=OPENAI_API_KEY)

# 모델은 필요하면 환경변수로 바꿔도 됨
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

app = FastAPI()

# CORS (기존 스타일 유지)
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


# =========================
# Helpers
# =========================
def _cut_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars]
    return s


def _to_response(detail: dict) -> BoardAiResponse:
    return BoardAiResponse(
        resultText=detail.get("content", "") or "",
        score=detail.get("score", None),
        modelName=detail.get("model"),
        tokensIn=detail.get("tokens_in"),
        tokensOut=detail.get("tokens_out"),
        tokensTotal=detail.get("tokens_total"),
        latencyMs=detail.get("latency_ms"),
    )


def _call_llm_text(full_prompt: str, max_tokens: int = 850) -> dict:
    """
    텍스트 LLM 호출 (summary/sentiment/write 용)
    chatbot.py 없이 OpenAI 직접 호출
    """
    try:
        t0 = time.perf_counter()

        r = client.chat.completions.create(
            model=CHAT_MODEL,
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
        used_model = getattr(r, "model", None) or CHAT_MODEL

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


def _call_llm_image(full_prompt: str, data_url: str, max_tokens: int = 350) -> dict:
    """
    이미지 포함 LLM 호출 (moderate-image 용)
    - 결과는 반드시 JSON 한 줄만 나오도록 프롬프트에서 강제
    """
    try:
        t0 = time.perf_counter()

        r = client.chat.completions.create(
            model=CHAT_MODEL,
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
        used_model = getattr(r, "model", None) or CHAT_MODEL

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
    # data:image/png;base64,... 형태면 prefix 제거
    if b64.startswith("data:") and "base64," in b64:
        b64 = b64.split("base64,", 1)[1].strip()

    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="imageBase64 decode failed")


def _make_data_url(image_bytes: bytes, content_type: str) -> str:
    ct = (content_type or "").strip().lower()
    if not ct.startswith("image/"):
        ct = "image/png"  # fallback
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{ct};base64,{b64}"


# =========================
# Routes
# =========================
@app.get("/")
def hello():
    return {"hello": "Board AI FastAPI (standalone)"}


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
    detail = _call_llm_text(full_prompt)
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
    detail = _call_llm_text(full_prompt)
    return _to_response(detail)


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
    detail = _call_llm_text(full_prompt)
    return _to_response(detail)


# =========================
# NEW: Image Moderation (LLM)
# =========================
@app.post("/board/moderate-image", response_model=BoardAiResponse)
def board_moderate_image(req: BoardAiImageRequest):
    image_bytes = _decode_base64_image(req.imageBase64)
    data_url = _make_data_url(image_bytes, req.contentType)

    filename = _cut_text(req.filename or "", 300)
    content_type = _cut_text(req.contentType or "", 100)

    # Java에서 단순 파싱하니까, resultText는 JSON "한 줄"만 나오게 강제
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
    detail = _call_llm_image(full_prompt, data_url, max_tokens=250)

    # resultText가 비면 실패로 처리
    if not detail.get("content"):
        raise HTTPException(status_code=502, detail="AI response empty")

    return _to_response(detail)


if __name__ == "__main__":
    uvicorn.run("boardaimain:app", host="0.0.0.0", port=8000, reload=True)
