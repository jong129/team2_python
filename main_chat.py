# main.py
import math
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

import uvicorn
from openai import OpenAI

client = OpenAI()

# =========================
# In-Memory Vector Store (RAG)
# =========================
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def make_context_from_hits(hits: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    chunks, total = [], 0
    for h in hits:
        t = h["text"].strip()
        if not t:
            continue
        piece = f"- {t}"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n".join(chunks)

# =========================
# RAG Prompt
# =========================
def build_rag_prompt(question: str, context: str) -> str:
    return f"""
너는 부동산 계약서와 등기부등본을 쉽게 설명하는 전문가다.

[규칙]
- 불필요한 인삿말 금지
- 참고 자료를 그대로 복붙하지 말 것
- 확실하지 않으면 확인 방법 제시
- 한국어로 답변

[참고 자료]
{context if context else "(없음)"}

[질문]
{question}

[답변]
""".strip()

# =========================
# OpenAI Helpers
# =========================
def create_embedding(text: str) -> List[float]:
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return r.data[0].embedding

def chat_answer(prompt: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 정확한 부동산 문서 분석 AI다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    return r.choices[0].message.content.strip()

def generate_title_from_messages(raw: str) -> str:
    """
    raw: 'user: ...\nassistant: ...' 같은 대화 요약 원문(앞부분 몇 개 메시지)
    return: 한국어 10~20자(최대 25자) 제목 1줄
    """
    prompt = f"""
아래는 한 세션의 대화 일부다. 이 대화의 핵심 주제를 반영하는 세션 제목을 한국어로 1줄만 만들어라.

[제약]
- 10~20자 권장, 최대 25자
- 따옴표(" ') / 마침표(.) / 이모지 사용 금지
- 접두어(예: "제목:", "요약:") 금지
- 너무 일반적인 말(예: "질문", "문의", "상담")만 쓰지 말 것
- 부동산/계약/등기부/보증금/특약/중도금/해지/위약금 등 핵심 키워드가 있으면 반영
- 반드시 제목만 출력

[대화]
{raw}
""".strip()

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 대화 내용을 한 줄 제목으로 요약하는 도우미다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=50,
    )

    title = (r.choices[0].message.content or "").strip()

    # 후처리: 따옴표/줄바꿈 제거, 길이 제한
    title = title.replace("\n", " ").replace("\r", " ").strip()
    title = title.replace('"', "").replace("'", "").replace(".", "").strip()
    if len(title) > 25:
        title = title[:25].strip()
    return title

