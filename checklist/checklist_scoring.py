# checklist/checklist_scoring.py

from typing import Dict, List
from pydantic import BaseModel
import json

from checklist.checklist_rag import rag_service


class ChecklistScoreItem(BaseModel):
    itemId: int
    title: str
    description: str

class ChecklistScoreRequest(BaseModel):
    items: List[ChecklistScoreItem]

class ChecklistScoreResult(BaseModel):
    itemId: int
    title: str
    importanceScore: float
    reason: str

class ChecklistScoreResponse(BaseModel):
    scores: List[ChecklistScoreResult]

class ChecklistScoringService:
    """
    ==================================================
    체크리스트 항목 중요도(위험도) 스코어링 서비스
    - PDF 기반 공공 가이드(RAG) 활용
    - 항목별 중요도 점수 산출 (0.0 ~ 1.0)
    - POST_A / POST_B 결정은 하지 않음
    ==================================================
    """

    def __init__(self, rag_service):
        """
        rag_service: ChecklistRagService 인스턴스
        """
        self.rag = rag_service
        self.llm = rag_service.llm

    # ==================================================
    # 1️⃣ 단일 체크리스트 항목 중요도 평가
    # ==================================================
    def score_item(self, item: Dict) -> Dict:
        """
        체크리스트 항목 1개의 중요도 점수를 평가한다.
        """

        query = f"""
        전세 계약 사기 예방 관점에서
        '{item.get("title")}' 항목을 이행하지 않았을 때
        발생할 수 있는 구체적인 위험 사례
        """

        # 🔍 PDF 기반 문맥 검색
        context = self.rag._retrieve_context(query)

        prompt = f"""
너는 전세 사기 예방을 위한
공공 가이드 문서를 분석하는 전문가다.

아래 문서를 근거로,
주어진 체크리스트 항목을 이행하지 않았을 때
발생할 수 있는 위험의 중요도를 평가하라.

평가 원칙:
- 반드시 문서 내용에 근거하여 판단
- 다른 항목에도 그대로 적용될 수 있는 일반적인 설명은 피할 것
- 이 항목을 누락했을 때 실제로 발생 가능한 문제를 중심으로 설명
- POST_A / POST_B, 합격/불합격 같은 판단은 하지 말 것
- JSON 외의 텍스트는 출력하지 말 것

중요도 점수 기준:
- 0.9 ~ 1.0 : 누락 시 즉각적이거나 심각한 피해 가능
- 0.7 ~ 0.8 : 매우 중요하며 강하게 권고되는 항목
- 0.4 ~ 0.6 : 중요하지만 상황에 따라 영향이 달라질 수 있음
- 0.1 ~ 0.3 : 보조적 확인 사항
- 0.0 : 문서에서 거의 언급되지 않거나 관련 없음

[공공 가이드 문서 ]
{context}

[체크리스트 항목]
- 제목: {item.get("title")}
- 설명: {item.get("description")}

반드시 아래 JSON 형식으로만 응답하라.

응답 형식:
{{
  "importanceScore": 0.0,
  "reason": "이 항목을 이행하지 않았을 때 발생 가능한 위험을 한 문장으로 설명"
}}
"""

        response = self.llm.invoke(prompt).content.strip()

        try:
            result = json.loads(response)
        except Exception:
            # ⚠️ 파싱 실패 시 보수적 기본값
            result = {
                "importanceScore": 0.5,
                "reason": "문서와의 연관성을 명확히 판단하지 못함"
            }

        return {
            "itemId": item.get("itemId"),
            "title": item.get("title"),
            "importanceScore": round(float(result.get("importanceScore", 0.5)), 2),
            "reason": result.get("reason", "")
        }

    # ==================================================
    # 2️⃣ 여러 항목 일괄 스코어링
    # ==================================================
    def score_items(self, items: list[Dict]) -> Dict:
        """
        여러 체크리스트 항목을 일괄 평가한다.
        """

        results = []

        for item in items:
            scored = self.score_item(item)
            results.append(scored)

        return {
            "scores": results
        }
        
    # ==================================================
    # 3️⃣ API 단위: 중요도 스코어링
    # ==================================================
    def score(self, req: ChecklistScoreRequest) -> ChecklistScoreResponse:
        """
        /checklist/ai/score 전용 엔트리포인트
        """

        items = [
            {
                "itemId": i.itemId,
                "title": i.title,
                "description": i.description,
            }
            for i in req.items
        ]

        result = self.score_items(items)

        return ChecklistScoreResponse(
            scores=result.get("scores", [])
        )


scoring_service = ChecklistScoringService(rag_service)