import json
from typing import List, Tuple, Dict, Any

from openai import OpenAI
from db import SessionLocal
from document.registry.risk.registry_risk_policy import get_active_policy
from document.registry.risk.registry_risk_rule import get_active_rules
from tool import logger

client = OpenAI()


def build_vision_content(user_text: str, images_b64: List[str]) -> List[dict]:
    content = [{"type": "text", "text": user_text}]
    for b64 in images_b64:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    return content
# -------------------------------------------------
# 위험도 점수 스케일링
# -------------------------------------------------
def scale_to_100(raw_score: int, rules) -> int:
    max_possible = sum(int(r.score) for r in rules)  # rules 전체 점수 합
    if max_possible <= 0:
        return 0
    scaled = int(round((raw_score / max_possible) * 100))
    return max(0, min(100, scaled))

# -------------------------------------------------
# 위험 점수 계산 (DB 룰 기반)
# -------------------------------------------------
def calculate_registry_risk_score(parsed_data: dict, rules) -> Tuple[int, List[str]]:
    total_score = 0
    reasons: List[str] = []

    for rule in rules:
        if rule.category == "임차권":
            lease = parsed_data.get("임차권", {})
            if rule.rule_key == "exists" and lease.get("exists"):
                total_score += rule.score
                reasons.append(rule.description)
            if rule.rule_key == "not_prior" and lease.get("is_prior") is False:
                total_score += rule.score
                reasons.append(rule.description)

        elif rule.category == "압류":
            arrest = parsed_data.get("압류", {})
            if rule.rule_key == "exists" and arrest.get("exists"):
                total_score += rule.score
                reasons.append(rule.description)
            if rule.rule_key == "multiple" and (arrest.get("count") or 0) >= 2:
                total_score += rule.score
                reasons.append(rule.description)           

        elif rule.category == "가압류":
            if parsed_data.get("가압류", {}).get("exists"):
                total_score += rule.score
                reasons.append(rule.description)

        elif rule.category == "근저당":
            if parsed_data.get("근저당", {}).get("exists"):
                total_score += rule.score
                reasons.append(rule.description)
    scaled = scale_to_100(total_score, rules)
    return scaled, reasons


# -------------------------------------------------
# REGISTRY 전용: Vision OCR + 구조화
# -------------------------------------------------
def parse_registry_info(images_b64: List[str]) -> Dict[str, Any]:
    system_prompt = """
너는 등기 및 권리관계 문서 이미지에서 정보를 추출해
오직 JSON 객체 1개만 출력하는 OCR 파서다.

절대 규칙:
- JSON 외 텍스트(설명, 문장, 주석, 마크다운, ```) 금지
- 키 누락, 추가, 이름 변경 금지
- 값은 JSON 값만 사용 (true/false/null/number/string/object/array)
- 불확실하거나 이미지에 없으면 null
- 추측, 법적 판단, 요약, 해석 금지
""".strip()

    user_prompt = """
아래 이미지에서 명시적으로 확인되는 사실만 추출하라.

반환은 반드시 아래 JSON 스켈레톤을 그대로 사용하라.
JSON 외의 어떤 문자도 출력하지 마라.

{
  "임차권": {"exists": false, "is_prior": null, "deposit": null},
  "압류": {"exists": false, "type": null, "count": null},
  "가압류": {"exists": false, "amount": null},
  "근저당": {"exists": false, "max_amount": null},
  "신탁": {"exists": false},
  "meta": {"uncertain_fields": []}
}

추가 규칙:
- exists / is_prior 는 true 또는 false만 사용
- 금액, 개수는 숫자만 사용 (원, 콤마, 문자 금지)
- type 은 "국세" | "지방세" | "기타" 중 하나만 사용, 불명확하면 null
- 확신이 낮은 필드는 meta.uncertain_fields 에 필드 경로 문자열로 추가
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_vision_content(user_prompt, images_b64)},
        ],
        temperature=0,
        max_tokens=1200,
    )

    raw = (resp.choices[0].message.content or "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Vision JSON 파싱 실패 (리페어 후)", extra={"raw": raw})
        raise ValueError("Vision AI JSON 파싱 실패")


# -------------------------------------------------
# 외부에서 호출할 함수: analyze_registry(images_b64)
# -------------------------------------------------
def analyze_registry(images_b64: List[str]) -> Tuple[int, List[str], str, Dict[str, Any]]:
    db = SessionLocal()
    try:
        policy = get_active_policy(db)
        if not policy:
            raise ValueError("활성 정책이 없습니다.")
        
        rules = get_active_rules(db, policy.id)
        parsed_data = parse_registry_info(images_b64)
        score, reasons = calculate_registry_risk_score(parsed_data, rules)
        return score, reasons, policy.version
    finally:
        db.close()
