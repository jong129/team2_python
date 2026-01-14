# document/document.py
import base64
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum  # ✅ sqlalchemy Enum 말고 파이썬 Enum

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from document.registry.registry import analyze_registry
from tool import logger

client = OpenAI()


# ====== DocType ======
class DocType(str, Enum):
    CONTRACT = "CONTRACT"
    REGISTRY = "REGISTRY"
    BUILDING = "BUILDING"
    UNKNOWN = "UNKNOWN"


@dataclass
class ClassifyResult:
    doc_type: DocType
    confidence: Optional[int] = None
    evidence: Optional[List[str]] = None
    raw_json: Optional[Dict[str, Any]] = None
    override_reason: Optional[str] = None


# ====== Rule keywords (강제 보정용) ======
BUILDING_FORCE_KEYWORDS = [
    "건축물대장",
    "일반건축물대장",
    "집합건축물대장",
    "건축물현황도",
    "위반건축물",
    "건축물대장 발급",
    "건축물대장 열람",
]

CONTRACT_HINT_KEYWORDS = [
    "임대차계약서",
    "전세계약서",
    "부동산 임대차",
    "계약금",
    "잔금",
    "특약사항",
    "중개대상물",
    "중개사무소",
]

REGISTRY_HARD = [
    "부동산 등기사항증명서",
    "등기사항증명서",
    "등기부등본",
    "갑구",
    "을구",
    "등기원인",
    "등기원인 및 기타사항",
    "대지권의 목적인 토지의 표시",
    "접수",
]


# -------------------------------------------------
# 이미지 로딩/인코딩
# -------------------------------------------------
def encode_image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_images_b64(image_path: str) -> List[str]:
    return [encode_image_to_b64(image_path)]


def _normalize_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _contains_any(text: str, keywords: List[str]) -> Optional[str]:
    t = _normalize_text(text)
    for kw in keywords:
        if kw in t:
            return kw
    return None


def override_doc_type(model_doc_type: DocType, evidence_text: str) -> Tuple[DocType, Optional[str]]:
    t = _normalize_text(evidence_text)

    hit = _contains_any(t, REGISTRY_HARD)
    if hit:
        return DocType.REGISTRY, f"REGISTRY 강제 키워드 감지: '{hit}'"

    hit = _contains_any(t, BUILDING_FORCE_KEYWORDS)
    if hit:
        return DocType.BUILDING, f"BUILDING 강제 키워드 감지: '{hit}'"

    hit = _contains_any(t, CONTRACT_HINT_KEYWORDS)
    if hit:
        return DocType.CONTRACT, f"CONTRACT 힌트 키워드 감지: '{hit}'"

    return model_doc_type, None


# -------------------------------------------------
# ✅ Vision 호출 함수 (OpenAI)
# -------------------------------------------------
def call_vision_fn_single(img_b64: str, system_prompt: str, user_prompt: str) -> str:
    """
    OpenAI Vision 호출.
    - responses API/ chat.completions 중 아무거나 써도 되는데,
      여기서는 가장 단순한 형태로 작성.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            },
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


# -------------------------------------------------
# 문서 분류 + override
# -------------------------------------------------
def classify_document_with_override(img_b64: str) -> ClassifyResult:
    system_prompt = """
너는 한국 부동산 관련 문서 이미지의 '문서 종류'만 분류하는 분류기다.
반드시 JSON 객체 1개만 출력한다.

절대 규칙:
- JSON 외 텍스트(설명/문장/주석/마크다운/```) 금지
- 키 추가/삭제/변경 금지
- 판단은 문서에 실제로 보이는 단서에만 근거
- 애매하면 UNKNOWN

특히 중요한 규칙(오분류 방지):
- "부동산 등기사항증명서", "등기원인", "접수", "갑구", "을구", "대지권의 목적인 토지의 표시"가 보이면 무조건 REGISTRY.
- 건물 구조/층별 면적 표만으로 BUILDING이라 단정하지 말 것(등기부 표제부에도 동일하게 나옴).
""".strip()

    user_prompt = """
아래 이미지가 어떤 문서인지 분류하라.

가능한 doc_type 값:
- CONTRACT  : 임대차계약서/전세계약서/부동산 임대차 계약 관련 문서
- REGISTRY  : 등기사항증명서(등기부등본)/등기 관련 권리관계 문서
- BUILDING  : 건축물대장/건축물현황도 등 건축물 관련 행정 문서
- UNKNOWN   : 단서 부족/애매

반드시 아래 JSON 스키마로만 출력:
{
  "doc_type": "CONTRACT|REGISTRY|BUILDING|UNKNOWN",
  "doc_confidence": 0-100,
  "doc_evidence": ["이미지에서 직접 확인한 단서 3~8개, 짧게"]
}
""".strip()

    raw = call_vision_fn_single(img_b64, system_prompt, user_prompt)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return ClassifyResult(doc_type=DocType.UNKNOWN, override_reason="LLM 응답에서 JSON을 찾지 못함")

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return ClassifyResult(doc_type=DocType.UNKNOWN, override_reason="LLM 응답 JSON 파싱 실패", raw_json={"raw": raw})

    # ✅ doc_type 정규화 (공백/소문자 방어)
    dt_raw = str(data.get("doc_type", "UNKNOWN")).strip().upper()
    model_type = DocType(dt_raw) if dt_raw in DocType._value2member_map_ else DocType.UNKNOWN

    confidence = data.get("doc_confidence")
    evidence_list = data.get("doc_evidence") or []
    if not isinstance(evidence_list, list):
        evidence_list = [str(evidence_list)]

    evidence_text = " ".join([str(x) for x in evidence_list])
    final_type, reason = override_doc_type(model_type, evidence_text)

    return ClassifyResult(
        doc_type=final_type,
        confidence=int(confidence) if isinstance(confidence, (int, float)) else None,
        evidence=evidence_list,
        raw_json=data,
        override_reason=reason,
    )


# -------------------------------------------------
# AI 설명 생성
# -------------------------------------------------
def generate_ai_explanation(score: int, reasons: List[str], policy_version: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
너는 전세계약 위험도 결과를 설명하는 AI다.
새로운 판단이나 점수 계산을 절대 하지 마라.

[정책 버전]
{policy_version}

[위험 점수]
{score}점

[판단 근거]
{chr(10).join(f"- {r}" for r in reasons) if reasons else "- (해당 없음)"}
""".strip()

    resp = llm.invoke([SystemMessage(content=prompt)])
    return resp.content


# -------------------------------------------------
# 최종 진입 함수
# -------------------------------------------------
def analyze_document(image_path: str) -> Dict[str, Any]:
    try:
        images_b64 = load_images_b64(image_path)
        img_b64 = images_b64[0]

        cls = classify_document_with_override(img_b64)
        doc_type = cls.doc_type

        logger.info("문서 타입=%s conf=%s evidence=%s override=%s",
                    doc_type, cls.confidence, cls.evidence, cls.override_reason)

        if doc_type == DocType.UNKNOWN:
            raise ValueError("문서 타입을 분류할 수 없습니다.")

        # 일단 지금은 전부 analyze_registry로 보내고 있는데,
        # 추후 doc_type별로 analyze_registry/analyze_contract/analyze_building으로 분기하면 됨.
        score, reasons, policy_version, parsed_data = analyze_registry(images_b64)

        explanation = generate_ai_explanation(score, reasons, policy_version)

        return {
            "doc_type": doc_type.value,  # ✅ Enum이면 .value로 문자열 내려주는 게 안전
            "policy_version": policy_version,
            "risk_score": score,
            "reasons": reasons,
            "ai_explanation": explanation,
            # 디버깅/추적용(원하면 빼도 됨)
            "doc_confidence": cls.confidence,
            "doc_evidence": cls.evidence,
            "override_reason": cls.override_reason,
            "parsed_data": parsed_data,
        }

    except Exception:
        logger.error("문서 분석 실패", exc_info=True, extra={"image_path": image_path})
        raise



# -------------------------------------------------
# 챗봇 연동
# -------------------------------------------------
def analyze_document_b64(img_b64: str) -> Dict[str, Any]:
    images_b64 = [img_b64]

    cls = classify_document_with_override(img_b64)
    doc_type = cls.doc_type

    if doc_type == DocType.UNKNOWN:
        raise ValueError("문서 타입을 분류할 수 없습니다.")

    score, reasons, policy_version, parsed_data = analyze_registry(images_b64)
    explanation = generate_ai_explanation(score, reasons, policy_version)

    return {
        "doc_type": doc_type.value,
        "policy_version": policy_version,
        "risk_score": score,
        "reasons": reasons,
        "ai_explanation": explanation,
        "doc_confidence": cls.confidence,
        "doc_evidence": cls.evidence,
        "override_reason": cls.override_reason,
        "parsed_data": parsed_data,
    }