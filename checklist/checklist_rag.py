# checklist/checklist_rag.py

from typing import List, Dict
import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document



class ChecklistRagService:
    """
    ==========================================
    전세 사기 예방 체크리스트 RAG 서비스
    - PDF 기반 지식 벡터화
    - 신규 체크리스트 항목 생성
    - PDF 기준(가이드라인) 추출
    ==========================================
    """

    def __init__(self, pdf_path: str, txt_path: str):
        self.pdf_path = pdf_path
        self.txt_path = txt_path
        self._init_vector_store()

    # ==================================================
    # 1️⃣ RAG 공통 초기화
    # ==================================================
    def _init_vector_store(self):
        """
        PDF + TXT → Chunk → VectorStore → Retriever → LLM
        """

        # 📘 PDF 로딩
        loader = PyPDFLoader(self.pdf_path)
        pdf_docs = loader.load()

        # 📄 TXT 로딩
        txt_docs = self._load_txt(self.txt_path)

        # 📚 문서 병합 (PDF + TXT 동급)
        all_docs = pdf_docs + txt_docs

        # ✂️ 문서 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(all_docs)

        # 🔢 임베딩
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # 🧠 벡터 스토어
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

        # 🔍 Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        # 🤖 LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        
    def _load_txt(self, txt_path: str) -> List[Document]:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        return [
            Document(
                page_content=text,
                metadata={"source": "CHECKLIST_TXT"}
            )
        ]

    

    # ==================================================
    # 2️⃣ 공통 Context 검색
    # ==================================================
    def _retrieve_context(self, query: str) -> str:
        """
        검색 쿼리에 맞는 PDF 문맥을 가져온다
        """
        docs = self.retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])

    # ==================================================
    # 3️⃣ 기존 기능: 신규 체크리스트 항목 생성
    # ==================================================
    def generate_new_items(
        self,
        base_items: List[str],
        phase: str
    ) -> Dict:
        """
        기존 체크리스트를 기준으로
        PDF 근거 기반 신규 항목 생성
        """

        # 🔍 검색 쿼리
        query = f"전세 계약 {phase} 단계에서 발생할 수 있는 사기 위험"
        context = self._retrieve_context(query)

        # 🧠 프롬프트
        prompt = f"""
너는 전세 사기 예방 전문가다.

아래는 현재 사용 중인 체크리스트 항목이다:
{base_items}

아래 문서를 참고하여,
- 기존 항목으로 커버되지 않는 위험 요소만 골라
- 신규 체크리스트 항목을 제안하라

규칙:
- 문서 내용에 근거한 항목만 생성
- 중복 항목 생성 금지
- JSON 외의 어떤 텍스트도 절대 출력하지 마라
- 응답은 반드시 {{ 로 시작하고 }} 로 끝나야 한다

응답 형식:
{{
  "new_items": [
    {{
      "title": "항목 제목",
      "description": "왜 필요한지 설명",
      "source": "PDF 근거 요약"
    }}
  ]
}}

문서:
{context}
"""

        response = self.llm.invoke(prompt).content

        try:
            return json.loads(response)
        except Exception:
            return {"new_items": []}

    # ==================================================
    # 4️⃣ 신규 기능: PDF 기준(가이드라인) 추출
    # ==================================================
    def extract_guidelines(self) -> Dict:
        """
        PDF에서 체크리스트 항목 생성의
        '정책/행동 기준'을 추출한다
        """

        query = """
        전세 계약 과정에서
        반드시 확인해야 하는
        사기 예방 핵심 행동 기준
        """

        context = self._retrieve_context(query)

        prompt = f"""
너는 전세 사기 예방 공공 가이드를 분석하는 전문가다.

아래 문서를 기준으로,
체크리스트 항목 생성의 근거가 될 수 있는
'행동 기준'만 추출하라.

규칙:
- 행동 단위로 정리
- 중요도가 높은 것 위주
- 중복 제거
- 일반 설명, 사례, 서론 제거
- JSON 외 텍스트 출력 금지

형식:
{{
  "guidelines": [
    {{
      "guideline_id": "REGISTRY_CHECK",
      "title": "등기부등본 재확인",
      "importance": "HIGH",
      "description": "왜 반드시 확인해야 하는지",
      "source": "PDF 근거 요약"
    }}
  ]
}}

문서:
{context}
"""

        response = self.llm.invoke(prompt).content

        try:
            return json.loads(response)
        except Exception:
            return {"guidelines": []}
          
    def explain_item_reason(
        self,
        guideline: Dict,
        user_stats: Dict,
        satisfaction: Dict,
        preview_item: Dict
    ) -> str:
        """
        AI 개선 체크리스트 항목이
        왜 추가/변경되었는지 설명을 생성한다.
        """

        prompt = f"""
너는 전세 사기 예방 체크리스트를 검토하는 관리자 보조 AI다.

아래 정보를 근거로,
"이 체크리스트 항목이 왜 추가되었는지"를
관리자가 빠르게 이해할 수 있도록 요약하라.

출력 규칙:
- 반드시 2문장 이내
- 각 문장은 25자 내외의 짧은 문장
- 불필요한 수식어, 배경 설명 금지
- 결론 위주로 설명
- 추측, 일반론, 과장 금지
- JSON, 마크다운, 번호, 불릿 사용 금지

설명 방식 예시:
"사용자 미이행 비율이 높고, 전세 사기 예방 가이드에서 중요하게 다루는 항목이다.
실제 계약 과정에서 문제가 반복되어 개선 항목으로 추가되었다."

[공공 가이드 기준]
- 항목명: {guideline.get("title")}
- 중요도: {guideline.get("importance")}
- 기준 설명: {guideline.get("description")}

[사용자 수행 통계]
- 완료 비율: {user_stats.get("doneRate")}
- 미완료 비율: {user_stats.get("notDoneRate")}
- 해당 없음 비율: {user_stats.get("notRequiredRate")}

[만족도 정보]
- 평균 점수: {satisfaction.get("avgScore")}
- 주요 부정 키워드: {satisfaction.get("negativeKeywords")}

[AI 개선 체크리스트 항목]
- 제목: {preview_item.get("title")}
- 설명: {preview_item.get("description")}
"""


        response = self.llm.invoke(prompt).content
        return response.strip()
