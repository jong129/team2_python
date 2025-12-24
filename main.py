from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from document import extract_document_info, analysis_document
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ★ 모든IP 허용
    allow_credentials=True, # 쿠키 기반 인증
    allow_methods=["*"],     # 필요에 따라 ["GET","POST"] 등으로 제한 가능
    allow_headers=["*"],     # 필요에 따라 ["Authorization","Content-Type"] 등으로 제한 가능
)


@app.get("/")  # http://localhost:8000
def hello():
    return {"hello": "FastAPI"}

@app.post("/document/analyze")
async def analyze_document(request: Request):
    data = await request.json()
    image_path = data.get("image_path")
    
    print('-> data:', data)
    # 여기에 문서 분석 로직 추가
    result=extract_document_info(image_path)
    parsed = json.loads(result)
    analysis = analysis_document(parsed)
    return analysis

if __name__ == "__main__":
    # main.py 파일명:FastAPI 객체 app 변수
    # host="0.0.0.0": 접속 가능 컴퓨터
    # reload=True: 소스 변경시 자동 재시작
    uvicorn.run("main:app", host="0.0.0.0", reload=True)


'''
(base) activate ai
(ai) python main.py
'''
