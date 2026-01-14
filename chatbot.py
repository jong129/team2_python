# chatbot.py
import os
import math
import json
import re
import time
import hashlib
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Chroma (Persistent)
import chromadb

load_dotenv()
client = OpenAI()

# =========================
# Chroma 설정
# =========================
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "real_estate")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dim
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
_chroma_col = _chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

# =========================
# Helpers
# =========================
def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().split())

def stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha1(_norm_text(text).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"

# =========================
# (호환용) cosine_similarity
# =========================
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

# =========================
# Chunking
# =========================
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    chunks = []
    start = 0
    n = len(t)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

# =========================
# Embedding (캐시)
# =========================
@lru_cache(maxsize=20000)
def _embed_cached(text: str) -> Tuple[float, ...]:
    t = _norm_text(text)
    if not t:
        return tuple()
    r = client.embeddings.create(model=EMBED_MODEL, input=t)
    return tuple(r.data[0].embedding)

def create_embedding(text: str) -> List[float]:
    return list(_embed_cached(text))

# =========================
# RAG Context
# =========================
def make_context_from_hits(hits: List[Dict[str, Any]], max_chars: int = 3500) -> str:
    chunks, total = [], 0
    for h in hits:
        t = (h.get("text") or "").strip()
        if not t:
            continue
        piece = f"- {t}"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n".join(chunks)
  
def classify_question(question: str) -> str:
    q = question.strip()

    SIMPLE_KEYWORDS = [
        "어디서", "어디", "방법", "어떻게", "발급", "신청",
        "비용", "수수료", "기간", "온라인", "사이트", "뭐"
    ]

    ANALYSIS_KEYWORDS = [
        "위험", "문제", "괜찮", "사기", "이상", "해석",
        "확인해야", "안전", "리스크", "문서", "체크리스트"
    ]

    if any(k in q for k in SIMPLE_KEYWORDS):
        return "simple"
    if any(k in q for k in ANALYSIS_KEYWORDS):
        return "analysis"

    return "analysis"  # 기본값

def build_simple_prompt(question: str) -> str:
    return f"""
사용자의 질문에 대해 간단하고 정확하게만 답변하라.

[규칙]
- 3~5줄 이내
- 불필요한 분석/위험도/체크리스트 금지
- 절차/장소/방법 위주
- 한국어

[질문]
{question}

[답변]
""".strip()
  

# =========================
# RAG Prompt
# =========================
def build_rag_prompt(question: str, context: str) -> str:
    return f"""
너는 부동산 계약서/등기부/확인서류를 쉽게 설명하는 전문가 AI다.

[출력 형식]
1) 결론(한 줄)
2) 핵심 요약(3~6줄)
3) 위험도: 낮음/중간/높음 + 근거
4) 근거(참고자료에서 중요한 부분만 요약)
5) 지금 해야 할 체크리스트(최대 7개)

[규칙]
- 불필요한 인삿말 금지
- 참고 자료를 그대로 복붙하지 말 것(요약/해석)
- 확실하지 않으면 확인 방법/필요 서류 제시
- 한국어로 답변

[참고 자료]
{context if context else "(없음)"}

[질문]
{question}

[답변]
""".strip()

# =========================
# Follow-up Questions (3)
# =========================
def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    return m.group(0) if m else ""

def build_followup_prompt(question: str, answer: str, hits: List[Dict[str, Any]]) -> str:
    # hits: chroma_search 결과 [{"id","text","meta","score"}, ...]
    lines = []
    for h in (hits or [])[:6]:
        cid = h.get("id")
        snip = (h.get("text") or "").replace("\n", " ")
        snip = snip[:200]
        lines.append(f"- chunkId={cid} | snippet={snip}")
    ref_text = "\n".join(lines) if lines else "(없음)"

    return f"""
당신은 사용자의 다음 질문(후속 질문) 3개를 추천하는 도우미입니다.

[사용자 질문]
{question}

[AI 답변]
{answer}

[근거 후보(Chunk)]
{ref_text}

요구사항:
- 후속 질문은 정확히 3개
- 각 질문은 40~90자 정도의 한국어
- 근거 chunkId를 직접 언급하진 말고, "근거를 확인"하도록 유도해도 좋음
- 출력은 반드시 JSON 하나만:
  {{"followUpQuestions":["...","...","..."]}}
- JSON 외의 텍스트는 절대 출력하지 마세요.
""".strip()

def fallback_followups() -> List[str]:
    return [
        "방금 답변을 한 줄 결론 + 3줄 요약으로 정리해줘",
        "이 답변에서 핵심 리스크 3가지만 뽑아줘",
        "내 상황에서 계약/의사결정 전에 확인할 체크리스트 5개 만들어줘",
    ]

def generate_followups(question: str, answer: str, hits: List[Dict[str, Any]]) -> List[str]:
    prompt = build_followup_prompt(question, answer, hits)
    raw = chat_answer(prompt)  # ✅ 이미 있는 chat_answer 재사용

    try:
        obj = json.loads(_extract_json(raw))
        arr = obj.get("followUpQuestions") or []
        arr = [x.strip() for x in arr if isinstance(x, str) and x.strip()]
        if len(arr) >= 3:
            return arr[:3]
    except Exception:
        pass

    return fallback_followups()



# =========================
# Chat Answer
# =========================
def chat_answer(prompt: str) -> str:
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "너는 정확한 부동산 문서 분석 AI다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=850,
    )
    return r.choices[0].message.content.strip()

def chat_answer_detail(prompt: str) -> Dict[str, Any]:
    t0 = time.perf_counter()

    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "너는 정확한 부동산 문서 분석 AI다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=850,
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)

    content = (r.choices[0].message.content or "").strip()

    usage = getattr(r, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
    tokens_out = getattr(usage, "completion_tokens", None) if usage else None
    tokens_total = getattr(usage, "total_tokens", None) if usage else None

    # ✅ 실제 사용 모델명: 응답에 있으면 그걸 쓰고, 없으면 CHAT_MODEL fallback
    used_model = getattr(r, "model", None) or CHAT_MODEL

    return {
        "content": content,
        "model": used_model,
        "tokens_in": int(tokens_in) if tokens_in is not None else None,
        "tokens_out": int(tokens_out) if tokens_out is not None else None,
        "tokens_total": int(tokens_total) if tokens_total is not None else (
            (int(tokens_in) if tokens_in is not None else 0) +
            (int(tokens_out) if tokens_out is not None else 0)
        ),
        "latency_ms": latency_ms,
    }
# =========================
# Title generator
# =========================
def generate_title_from_messages(raw: str) -> str:
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
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "너는 대화 내용을 한 줄 제목으로 요약하는 도우미다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=50,
    )

    title = (r.choices[0].message.content or "").strip()
    title = title.replace("\n", " ").replace("\r", " ").strip()
    title = title.replace('"', "").replace("'", "").replace(".", "").strip()
    if len(title) > 25:
        title = title[:25].strip()
    return title or "새 대화"

    


# =========================
# Chroma: add & search
# =========================
def chroma_add_docs(
    docs: List[Dict[str, Any]],
    chunk: bool = True,
    chunk_size: int = 900,
    overlap: int = 120,
) -> int:
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    embeddings: List[List[float]] = []

    inserted = 0

    for d in docs:
        base_id = d.get("id") or "doc"
        text = d.get("text") or ""

        # ✅ meta는 복사본으로 + 필터키는 문자열 통일 권장
        meta = dict(d.get("meta") or {})
        if "user_id" in meta and meta["user_id"] is not None:
            meta["user_id"] = str(meta["user_id"])
        if "doc_id" in meta and meta["doc_id"] is not None:
            meta["doc_id"] = str(meta["doc_id"])
        if "doc_type" in meta and meta["doc_type"] is not None:
            meta["doc_type"] = str(meta["doc_type"])
        if "stage" in meta and meta["stage"] is not None:
            meta["stage"] = str(meta["stage"])

        pieces = [text]
        if chunk:
            pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for i, p in enumerate(pieces):
            p = (p or "").strip()
            if not p:
                continue

            cid = stable_id(base_id, f"{i}:{p}")
            ids.append(cid)
            documents.append(p)

            # ✅ chunk마다 meta dict 복사해서 넣기(안전)
            metadatas.append(dict(meta))

            embeddings.append(create_embedding(p))
            inserted += 1

    if ids:
        _chroma_col.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return inserted


def chroma_search(
    query_embedding: List[float],
    top_k: int = 5,
    doc_type: Optional[str] = None,
    stage: Optional[str] = None,
    user_id: Optional[str] = None,   # ✅ 추가
    doc_id: Optional[str] = None,    # ✅ 추가
) -> List[Dict[str, Any]]:
    """
    return hits: [{"id","text","meta","score"}...]
    score는 (1 - distance) 형태로 근사 (높을수록 유사)
    """
    where = {}

    if doc_type is not None:
        where["doc_type"] = str(doc_type)
    if stage is not None:
        where["stage"] = str(stage)

    # ✅ B안: 특정 사용자/특정 문서로 제한
    if user_id is not None:
        where["user_id"] = str(user_id)
    if doc_id is not None:
        where["doc_id"] = str(doc_id)

    if not where:
        where = None

    include = ["documents", "metadatas", "distances"]
    res = _chroma_col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=include,
    )

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: List[Dict[str, Any]] = []
    for _id, doc, meta, dist in zip(ids, docs, metas, dists):
        score = float(1.0 - dist) if dist is not None else 0.0
        hits.append({
            "id": _id,
            "text": doc,
            "meta": meta or {},
            "score": score
        })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits
