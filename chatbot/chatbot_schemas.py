# chatbot_schemas.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

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
    
    user_id: Optional[str] = None
    doc_id: Optional[str] = None
    doc_type: Optional[str] = None   # 예: "contract" / "registry"
    