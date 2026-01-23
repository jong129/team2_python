from fastapi import HTTPException

from .schemas import BoardAiRequest, BoardAiImageRequest, BoardAiResponse
from .utils import cut_text, decode_base64_image, make_data_url
from .prompts import (
    build_summary_prompt,
    build_sentiment_prompt,
    build_write_prompt,
    build_moderate_image_prompt,
)
from .llm import call_llm_text, call_llm_image

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

def summary(req: BoardAiRequest) -> BoardAiResponse:
    title = cut_text(req.title, 500)
    content = cut_text(req.content, req.max_chars if req.truncate else 0)

    full_prompt = build_summary_prompt(req.prompt, title, content)
    detail = call_llm_text(full_prompt)
    return _to_response(detail)

def sentiment(req: BoardAiRequest) -> BoardAiResponse:
    title = cut_text(req.title, 500)
    content = cut_text(req.content, req.max_chars if req.truncate else 0)

    full_prompt = build_sentiment_prompt(req.prompt, title, content)
    detail = call_llm_text(full_prompt)
    return _to_response(detail)

def write(req: BoardAiRequest) -> BoardAiResponse:
    title = cut_text(req.title, 500)
    content = cut_text(req.content, req.max_chars if req.truncate else 0)

    if not title and not content:
        raise HTTPException(status_code=400, detail="title/content 둘 중 하나는 필요합니다.")

    full_prompt = build_write_prompt(req.prompt, title, content)
    detail = call_llm_text(full_prompt)
    return _to_response(detail)

def moderate_image(req: BoardAiImageRequest) -> BoardAiResponse:
    image_bytes = decode_base64_image(req.imageBase64)
    data_url = make_data_url(image_bytes, req.contentType)

    filename = cut_text(req.filename or "", 300)
    content_type = cut_text(req.contentType or "", 100)

    full_prompt = build_moderate_image_prompt(req.prompt, filename, content_type)
    detail = call_llm_image(full_prompt, data_url, max_tokens=250)

    if not detail.get("content"):
        raise HTTPException(status_code=502, detail="AI response empty")

    return _to_response(detail)
