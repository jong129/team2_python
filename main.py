from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from document.document import analyze_document
from tool import logger
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    image_path: str


@app.post("/document/analyze")
async def analyze_document_endpoint(req: AnalyzeRequest):
    try:
        result = analyze_document(req.image_path)
        return result
    except Exception as e:
        logger.error("문서 분석 실패", exc_info=e)
        raise HTTPException(status_code=500, detail="문서 분석 실패")
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
