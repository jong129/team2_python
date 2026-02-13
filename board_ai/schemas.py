from pydantic import BaseModel, Field
from typing import Optional

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
