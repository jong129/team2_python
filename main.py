# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import time
import base64
import binascii
import traceback
import uvicorn
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from typing import Optional
from board_ai.schemas import BoardAiRequest, BoardAiImageRequest, BoardAiResponse
from board_ai import service as board_ai_service

from tool import logger
from document.document import analyze_document, analyze_document_b64
from chatbot.chatbot import (
    make_context_from_hits, build_rag_prompt, create_embedding, chat_answer, chat_answer_detail,
    generate_title_from_messages, chroma_add_docs, chroma_search, classify_question,
    build_simple_prompt, generate_followups)
from chatbot.chatbot_schemas import (
    EmbeddingRequest, EmbeddingResponse, IngestDoc, IngestRequest, AskRequest, AskResponse,
    RagReference, TitleRequest, TitleResponse, AnalyzeRequest)

# =========================
# Checklist AI Services
# =========================
from checklist.checklist_rag import (
    rag_service,
    ChecklistAiPreviewRequest,
    ChecklistAiPreviewResponse,
    ChecklistImproveSummaryRequest,
    ChecklistImproveSummaryResponse,
)
from checklist.checklist_scoring import (
    scoring_service,
    ChecklistScoreRequest,
    ChecklistScoreResponse,
)

from checklist.checklist_review import (
    review_service,
    PostChecklistReviewRequest,
)

from checklist.checklist_summary import (
    ChecklistSummaryRequest,
    ChecklistSummaryResponse,
    summarize as summarize_checklist_service,
)


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
# Routes
# =========================
@app.get("/")
def hello():
    return {"hello": "FastAPI", "store": "chroma"}

# 문서 이미지 분석 : 이미지 경로 또는 base64를 받아서 텍스트/정보를 추출(등기부등본, 계약서 등)
@app.post("/document/analyze")
async def analyze_document_endpoint(req: AnalyzeRequest):
    try:
        # 1) 분석 수행
        if req.image_b64:
            result = analyze_document_b64(req.image_b64)
        elif req.image_path:
            result = analyze_document(req.image_path)
        else:
            raise HTTPException(status_code=400, detail="image_b64 또는 image_path 필요")

        # 2) doc_id 없으면 만들어서라도 넣기 (없으면 RAG 필터 못 씀)
        user_id = str(req.user_id) if req.user_id is not None else "anonymous"
        doc_id = str(req.doc_id) if req.doc_id is not None else f"tmp-{int(time.time())}"

        doc_type = result.get("doc_type", req.doc_type or "UNKNOWN")
        risk_score = result.get("risk_score")
        reasons = result.get("reasons") or []
        ai_explanation = result.get("ai_explanation") or ""

        # 3) Chroma 저장(중요: return 전에!)
        chroma_add_docs([{
            "id": f"doc-{user_id}-{doc_id}",
            "text": f"""문서 유형: {doc_type}
위험 점수: {risk_score}
위험 사유:
{chr(10).join(reasons)}

AI 설명:
{ai_explanation}
""",
            "meta": {
                "doc_type": str(doc_type),
                "user_id": str(user_id),
                "doc_id": str(doc_id),
                "stage": "analysis",
            },
        }])

        # 4) 프론트가 doc_id를 저장할 수 있게 결과에도 내려주기(추천)
        result["user_id"] = user_id
        result["doc_id"] = doc_id

        return result

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
    t0 = time.perf_counter()

    # 요청 들어온 값 로그(필터가 제대로 들어오는지 확인)
    logger.info(
        f"[ASK IN] qLen={len(req.question or '')} "
        f"top_k={req.top_k} doc_type={req.doc_type} stage={req.stage} user_id={req.user_id} doc_id={req.doc_id} "
        f"ctxLen={(len(req.context) if req.context else 0)}"
    )

    try:
        # 1) 질문 분류
        q_type = classify_question(req.question)

        # ✅ doc_id(또는 context)가 있으면 RAG 강제
        if req.doc_id or (req.context and req.context.strip()):
            q_type = "rag"

        # 2-1) simple
        if q_type == "simple":
            prompt = build_simple_prompt(req.question)
            detail = chat_answer_detail(prompt)

            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(f"[ASK OUT] type=simple latencyMs={latency_ms} model={detail.get('model')}")

            return AskResponse(
                answer=detail["content"],
                references=[],
                followUpQuestions=[
                    "이 내용을 3줄로 더 쉽게 설명해줘",
                    "이걸 하려면 준비물/서류가 뭐가 필요해?",
                    "내 상황에서 주의할 점 3가지만 알려줘"
                ],
                model=detail.get("model"),
                tokensIn=detail.get("tokens_in"),
                tokensOut=detail.get("tokens_out"),
                tokensTotal=detail.get("tokens_total"),
                latencyMs=detail.get("latency_ms") if detail.get("latency_ms") is not None else latency_ms,
            )

        # 2-2) rag
        q_emb = create_embedding(req.question)

        hits = chroma_search(
            query_embedding=q_emb,
            top_k=req.top_k,
            doc_type=req.doc_type,
            stage=req.stage,
            user_id=req.user_id,
            doc_id=req.doc_id,
        )

        # ⭐ hits 0건이면 이 시점에서 바로 원인 파악 가능
        logger.info(
            f"[ASK RAG] hits={len(hits)} "
            f"filters(doc_type={req.doc_type}, stage={req.stage}, user_id={req.user_id}, doc_id={req.doc_id})"
        )

        rag_context = make_context_from_hits(hits)

        merged_context = ""
        if req.context:
            merged_context += "=== [세션 참고자료] ===\n" + req.context.strip() + "\n\n"
        merged_context += "=== [RAG 검색자료] ===\n" + (rag_context.strip() if rag_context else "(없음)")

        prompt = build_rag_prompt(req.question, merged_context)
        detail = chat_answer_detail(prompt)
        answer = detail["content"]

        references = [
            RagReference(
                chunkId=h["id"],
                title=(h.get("meta") or {}).get("title"),
                snippet=(h.get("text") or "")[:200],
                score=float(h.get("score", 0.0)),
                rankNo=i + 1
            )
            for i, h in enumerate(hits)
        ]

        followups = generate_followups(req.question, answer, hits)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            f"[ASK OUT] type=rag latencyMs={latency_ms} model={detail.get('model')} "
            f"tokens_in={detail.get('tokens_in')} tokens_out={detail.get('tokens_out')} total={detail.get('tokens_total')}"
        )

        return AskResponse(
            answer=answer,
            references=references,
            followUpQuestions=followups,
            model=detail.get("model"),
            tokensIn=detail.get("tokens_in"),
            tokensOut=detail.get("tokens_out"),
            tokensTotal=detail.get("tokens_total"),
            latencyMs=detail.get("latency_ms") if detail.get("latency_ms") is not None else latency_ms,
        )

    except HTTPException:
        # 이미 status_code 있는 예외는 그대로 올림
        raise

    except Exception as e:
        # ✅ 여기서 traceback까지 남겨야 원인이 100% 잡힘
        tb = traceback.format_exc()
        logger.error(f"[ASK ERROR] {e}\n{tb}")

        # Spring에서 e.getResponseBodyAsString()으로 detail을 볼 수 있게 detail에 넣어줌
        raise HTTPException(
            status_code=500,
            detail=f"/ask failed: {type(e).__name__}: {str(e)}"
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
    return summarize_checklist_service(req)


@app.post("/checklist/ai/preview", response_model=ChecklistAiPreviewResponse)
def checklist_ai_preview(req: ChecklistAiPreviewRequest):
    return rag_service.preview(req)


@app.post("/checklist/ai/improve/summary", response_model=ChecklistImproveSummaryResponse)
def checklist_ai_improve_summary(req: ChecklistImproveSummaryRequest):
    return rag_service.improve_summary(req)


@app.post("/checklist/ai/score", response_model=ChecklistScoreResponse)
def checklist_score(req: ChecklistScoreRequest):
    return scoring_service.score(req)


@app.post("/checklist/post/review")
def review_post_checklist(req: PostChecklistReviewRequest):
    return review_service.review(req)


# =========================
# Board AI Routes (from boardaimain.py)
# =========================
@app.post("/board/summary", response_model=BoardAiResponse)
def board_summary(req: BoardAiRequest):
    return board_ai_service.summary(req)

@app.post("/board/sentiment", response_model=BoardAiResponse)
def board_sentiment(req: BoardAiRequest):
    return board_ai_service.sentiment(req)

@app.post("/board/write", response_model=BoardAiResponse)
def board_write(req: BoardAiRequest):
    return board_ai_service.write(req)

@app.post("/board/moderate-image", response_model=BoardAiResponse)
def board_moderate_image(req: BoardAiImageRequest):
    return board_ai_service.moderate_image(req) 

# =========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
