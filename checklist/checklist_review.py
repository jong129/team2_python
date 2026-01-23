# checklist/checklist_review.py
from typing import List, Dict
import json
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel



class PostChecklistReviewItem(BaseModel):
    itemId: int
    title: str
    description: str


class PostChecklistReviewRequest(BaseModel):
    total: int
    done: int
    notDoneItems: List[PostChecklistReviewItem]


class ChecklistReviewService:
    """
    ==================================================
    POST 체크리스트 진행 상태 리뷰 서비스
    - 미완료(NOT_DONE) 항목만 대상
    - PDF(RAG) + 중요도 스코어링 결과 기반
    - 사용자용 후속 조치 안내 생성
    ==================================================
    """

    def __init__(self, scoring_service):
        self.scoring = scoring_service
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def review_post_status(
        self,
        not_done_items: List[Dict],
        total: int,
        done: int
    ) -> Dict:
        """
        POST 체크리스트 현재 상태 리뷰 생성
        """

        not_done = len(not_done_items)

        if not not_done_items:
          return {
              "totalCount": total,
              "doneCount": done,
              "notDoneCount": not_done,
              "summary": message,
              "items": review_items
          }

        # 1️⃣ 중요도 스코어링 (PDF 근거)
        score_result = self.scoring.score_items(not_done_items)
        scores = score_result.get("scores", [])

        # 중요도 내림차순 정렬
        scores = sorted(
            scores,
            key=lambda x: x["importanceScore"],
            reverse=True
        )

        # 2️⃣ 상위 항목만 사용자 리뷰 대상으로 (최대 5개)
        top_items = scores[:5]

        # 3️⃣ 사용자 메시지 생성
        message = self._build_message(not_done, total)

        # 4️⃣ 항목별 후속 조치 문장 생성
        review_items = []
        for s in top_items:
            action = self._build_action(s["title"], s["reason"])
            review_items.append({
                "itemId": s["itemId"],
                "title": s["title"],
                "importanceScore": s["importanceScore"],
                "reason": s["reason"],
                "action": action
            })

        return {
            "total": total,
            "done": done,
            "notDone": not_done,
            "message": message,
            "items": review_items
        }

    # ==================================================
    # 내부 헬퍼
    # ==================================================
    def _build_message(self, not_done: int, total: int) -> str:
        """
        전체 상태 요약 문장
        """
        return (
            f"전체 {total}개 항목 중 {not_done}개가 아직 확인되지 않았습니다. "
            "아래 항목을 중심으로 추가 점검을 권장드립니다."
        )

    def _build_action(self, title: str, reason: str) -> str:
        """
        항목별 후속 조치 문장 생성
        """

        prompt = f"""
너는 전세 계약 사후 점검을 돕는 안내 AI다.

아래 정보를 근거로,
사용자가 다음에 취하면 좋은
'구체적이고 실행 가능한 후속 조치'를
한 문장으로 작성하라.

규칙:
- 과장, 공포 표현 금지
- 법적 판단, 계약 결론 제시 금지
- 반드시 안내형 문장
- JSON, 번호, 불릿 사용 금지

[미완료 항목]
- 제목: {title}

[위험 설명]
- {reason}
"""

        response = self.llm.invoke(prompt).content.strip()
        # 안전망
        if not response:
            return "관련 자료를 확인하고 필요한 후속 조치를 진행해 주세요."

        return response
