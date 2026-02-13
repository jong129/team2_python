# checklist/checklist_review.py
from typing import List, Dict
import json
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from checklist.checklist_scoring import scoring_service


class PostChecklistReviewItem(BaseModel):
    itemId: int
    title: str
    description: str


class PostChecklistReviewRequest(BaseModel):
    total: int
    done: int
    notDoneItems: List[PostChecklistReviewItem]

class PostChecklistSummaryItem(BaseModel):
    itemId: int
    title: str
    description: str
    status: str  # DONE / NOT_REQUIRED


class PostChecklistSummaryRequest(BaseModel):
    total: int
    done: int
    completedItems: List[PostChecklistSummaryItem]



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

        # not_done 항목이 없는 경우
        not_done = len(not_done_items)

        if not not_done_items:
          return {
              "totalCount": total,
              "doneCount": done,
              "notDoneCount": 0,
              "summary": "모든 항목이 확인되었습니다.",
              "items": []
          }

        # 1️⃣ 중요도 스코어링 (PDF 근거)
        score_result = self.scoring.score_items(not_done_items)
        scores = score_result.get("scores", [])

        if not scores:
            return {
                "totalCount": total,
                "doneCount": done,
                "notDoneCount": not_done,
                "summary": "일부 항목이 확인되지 않았으나, 중요도 분석에 필요한 정보가 부족합니다.",
                "items": []
            }

        # 중요도 내림차순 정렬
        scores = sorted(
            scores,
            key=lambda x: x["importanceScore"],
            reverse=True
        )

        # 2️⃣ 상위 항목만 사용자 리뷰 대상으로 (최대 5개)
        scores = [
            s for s in scores
            if s.get("importanceScore", 0) >= 0.3
        ]

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
            "totalCount": total,
            "doneCount": done,
            "notDoneCount": not_done,
            "summary": message,
            "items": review_items
        }

    def summarize_post_completed(
        self,
        completed_items: List[Dict],
        total: int,
        done: int
    ) -> Dict:
        """
        POST 체크리스트 완료 후 요약 생성
        - DONE / NOT_REQUIRED 기준
        - 경고 ❌
        - 유지·관리 가이드 ⭕
        """

        if not completed_items:
            return {
                "totalCount": total,
                "doneCount": done,
                "summary": "체크리스트가 완료되었으며, 추가로 안내할 사항은 없습니다.",
                "guides": []
            }

        prompt = f"""
너는 전세 계약 이후 사용자를 돕는 안내 AI다.

다음은 사용자가 사후 체크리스트를 모두 완료한 결과다.
이 정보를 바탕으로,
앞으로 보증금과 권리를 안전하게 유지하기 위한
'실천 중심의 가이드 요약'을 작성하라.

규칙:
- 경고, 공포, 위협 표현 금지
- 이미 완료한 행동을 존중하는 어조
- 법적 판단, 계약 결론 제시 금지
- 최대 3문장
- 안내형 문장 사용
- JSON 외 텍스트 출력 금지

완료된 항목:
{json.dumps(completed_items, ensure_ascii=False)}

출력 형식:
{{
  "summary": "전체 요약 문장",
  "guides": [
    "이후에 유의할 사항 1",
    "이후에 유의할 사항 2"
  ]
}}
"""

        response = self.llm.invoke(prompt).content.strip()

        try:
            return json.loads(response)
        except Exception:
            # 🔒 fallback
            return {
                "totalCount": total,
                "doneCount": done,
                "summary": "사후 점검이 정상적으로 완료되었습니다.",
                "guides": [
                    "계약 관련 서류를 안전하게 보관해 주세요.",
                    "추후 변동 사항 발생 시 다시 한 번 확인해 주세요."
                ]
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
    
    # ==================================================
    # 2️⃣ API 단위: POST 체크리스트 리뷰
    # ==================================================
    def review(
        self,
        req: PostChecklistReviewRequest
    ) -> Dict:
        """
        /checklist/post/review 전용 엔트리포인트
        """

        # 1️⃣ NOT_DONE 항목 변환 (scoring_service 입력 형식)
        not_done_items = [
            {
                "itemId": item.itemId,
                "title": item.title,
                "description": item.description,
            }
            for item in req.notDoneItems
        ]

        # 2️⃣ 리뷰 생성
        return self.review_post_status(
            not_done_items=not_done_items,
            total=req.total,
            done=req.done
        )
    
    def summarize_completed(
        self,
        req: PostChecklistSummaryRequest
    ) -> Dict:
        """
        /checklist/post/summary 전용 엔트리포인트
        """

        completed_items = [
            {
                "itemId": item.itemId,
                "title": item.title,
                "description": item.description,
                "status": item.status
            }
            for item in req.completedItems
        ]

        return self.summarize_post_completed(
            completed_items=completed_items,
            total=req.total,
            done=req.done
        )



review_service = ChecklistReviewService(scoring_service)