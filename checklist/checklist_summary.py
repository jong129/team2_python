# checklist/checklist_summary.py
from typing import List, Dict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os
import json
import re
from openai import OpenAI


class ChecklistSummaryRequest(BaseModel):
    templateId: int
    comments: List[str]

class ChecklistSummaryResponse(BaseModel):
    positive: List[str]
    negative: List[str]
    suggestions: List[str]
    
# =========================
# PRE 위험 설명 (사전 체크리스트)
# =========================

class PreRiskExplanationRequest(BaseModel):
    riskScoreSum: float
    reasons: List[str]


class PreRiskExplanationResponse(BaseModel):
    summary: str
    actions: List[str]

    
def is_meaningful_comment(comment: str) -> bool:
    """
    의미 없는 / 테스트용 / 운영 메타 코멘트 필터링
    """
    c = comment.strip()

    # 1️⃣ 길이 기반
    if len(c) < 5:
        return False

    # 2️⃣ 반복 문자
    if re.fullmatch(r"(.)\1{3,}", c):
        return False

    # 3️⃣ 자모만 있는 경우
    if re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣ]+", c):
        return False

    # 4️⃣ 완전 무의미 단어
    meaningless_exact = {
        "test", "asdf", "qwer", "...", "???", "!!!"
    }
    if c.lower() in meaningless_exact:
        return False

    # 5️⃣ 🔥 개발/운영/테스트 메타 코멘트 차단
    meaningless_keywords = [
        "테스트", "test", "테스트중", "확인",
        "개발", "개발중", "디버그",
        "ai", "분기", "로직", "api",
        "요약", "완성", "확인용",
        "집에서", "회사에서"
    ]

    lowered = c.lower()
    if any(k in lowered for k in meaningless_keywords):
        return False

    return True

def is_summary_worthy_comment(comment: str) -> bool:
    """
    이 코멘트가 '사용자 경험/개선 요약'에
    실질적으로 가치가 있는지 LLM으로 판별
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
다음 사용자 코멘트가
체크리스트 사용 경험, 만족/불만, 개선점 파악에
의미 있는 정보인지 판단하라.

다음 유형은 false:
- 테스트/개발/확인 목적
- 의미 없는 감상
- 구체적 경험이 없는 짧은 평가

반드시 true 또는 false 중 하나만 출력하라.

코멘트:
{comment}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 제품 UX 분석 전문가다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        result = response.choices[0].message.content.strip().lower()
        return result == "true"

    except Exception:
        # 🔒 실패 시 보수적으로 제외
        return False


def call_llm_for_summary(comments: List[str]) -> dict:
    """
    만족도 코멘트 요약
    """
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
다음은 체크리스트 사용자 만족도 코멘트 목록이다.

반드시 JSON으로만 응답하라.
설명, 마크다운, 코드블록은 절대 포함하지 마라.

형식:
{{
  "positive": ["..."],
  "negative": ["..."],
  "suggestions": ["..."]
}}

코멘트:
{chr(10).join(comments)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 체크리스트 UX 분석 전문가다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    try:
        result = json.loads(response.choices[0].message.content.strip())
        if not result.get("suggestions"):
            result["suggestions"] = ["특이 제안 없음."]
        return result
    except Exception:
        return {
            "positive": [],
            "negative": [],
            "suggestions": ["요약 생성 실패"]
        }
    


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

# ==================================================
# API 전용 서비스 인스턴스
# ==================================================
summary_service = ChecklistSummaryService()


def summarize(req: ChecklistSummaryRequest) -> ChecklistSummaryResponse:
    """
    /checklist/summary 전용 엔트리포인트
    """

    # 1️⃣ 규칙 기반 필터
    rule_filtered = [
        c for c in req.comments
        if is_meaningful_comment(c)
    ]

    # 2️⃣ LLM 기반 요약 가치 판별
    llm_filtered = [
        c for c in rule_filtered
        if is_summary_worthy_comment(c)
    ]

    if not llm_filtered:
        return ChecklistSummaryResponse(
            positive=[],
            negative=[],
            suggestions=["요약할 만한 사용자 경험 코멘트가 충분하지 않습니다."]
        )

    result = call_llm_for_summary(llm_filtered)

    return ChecklistSummaryResponse(
        positive=result.get("positive", []),
        negative=result.get("negative", []),
        suggestions=result.get("suggestions", []),
    )


def explain_pre_risk(req: PreRiskExplanationRequest) -> PreRiskExplanationResponse:
    """
    /checklist/pre/risk/explanation 전용 엔트리포인트
    - PRE 체크리스트 위험 요약
    - 이미 scoring에서 선별된 reason만 사용
    """

    result = summary_service.summarize_pre_result(
        top_reasons=req.reasons,
        max_lines=3
    )

    return PreRiskExplanationResponse(
        summary=result["summary"],
        actions=result["actions"]
    )
