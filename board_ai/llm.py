import os
import time
from fastapi import HTTPException
from openai import OpenAI

def call_llm_text(full_prompt: str, max_tokens: int = 850) -> dict:
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

def call_llm_image(full_prompt: str, data_url: str, max_tokens: int = 350) -> dict:
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
