# checklist/checklist_summary.py
from typing import List, Dict
from langchain_openai import ChatOpenAI
import os
import json


class ChecklistSummaryService:
    """
    사전 체크리스트 결과 요약 전용 서비스
    - RAG/Scoring에서 선별된 상위 reason을
      사용자용 요약 문장으로 변환
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def summarize_pre_result(
        self,
        top_reasons: List[str],
        max_lines: int = 3
    ) -> Dict:
        """
        사전 체크리스트 요약 생성
        - 이미 중요도로 선별된 reason만 사용
        """

        if not top_reasons:
            return {
                "summary": "사전 점검 결과, 특별히 주의가 필요한 항목은 확인되지 않았습니다.",
                "actions": ["계약 전 기본 사항을 한 번 더 점검해 주세요."]
            }

        prompt = f"""
너는 전세 계약 사전 점검 결과를
일반 사용자에게 설명하는 요약 AI다.

규칙:
- 반드시 아래 제공된 내용만 바탕으로 요약
- 최대 {max_lines}문장
- 과장, 단정, 공포 표현 금지
- 계약 판단이나 결론 제시 금지
- 조심스럽고 안내형 어조 유지
- JSON 외 텍스트 출력 금지

[확인이 필요한 주요 사유]
{chr(10).join(top_reasons)}

출력 형식:
{{
  "summary": "요약 문장",
  "actions": [
    "권장 행동 1",
    "권장 행동 2"
  ]
}}
"""

        response = self.llm.invoke(prompt).content.strip()

        try:
            return json.loads(response)
        except Exception:
            # LLM 실패 시 보수적 fallback
            return {
                "summary": "사전 점검 결과, 일부 항목에 대해 추가 확인이 권장됩니다.",
                "actions": [
                    "계약 전 주요 점검 항목을 다시 확인해 주세요.",
                    "필요하다면 전문가의 도움을 받아 검토해 보세요."
                ]
            }
