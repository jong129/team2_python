import json
from typing import List, Tuple, Dict, Any

from openai import OpenAI
from db import SessionLocal
from document.constract.risk.contract_risk_policy import get_active_policy
from document.constract.risk.contract_risk_rule import get_active_rules
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
# 위험 점수 계산 (DB 룰 기반) - CONTRACT
# -------------------------------------------------
def calculate_contract_risk_score(parsed_data: dict, rules) -> Tuple[int, List[str]]:
    """
    parsed_data 구조(권장):
    {
      "rules": {
        "waive_rights": {"flag": true, "evidence": [...]},
      },
      "extracted": {...}
    }
    """
    total_score = 0
    reasons: List[str] = []

    rules_node = (parsed_data or {}).get("rules", {})

    for rule in rules:
        # rule.rule_key 예: "waive_rights"
        node = rules_node.get(rule.rule_key, {})
        flag = bool(node.get("flag", False))
        if flag:
            total_score += int(rule.score)
            reasons.append(rule.description)

    return total_score, reasons


# -------------------------------------------------
# CONTRACT 전용: Vision OCR + 구조화
# -------------------------------------------------
def parse_contract_info(images_b64: List[str]) -> Dict[str, Any]:
    system_prompt = """
너는 한국 임대차계약서(전세/월세/반전세) 이미지에서 정보를 추출해
오직 JSON 객체 1개만 출력하는 OCR/구조화 파서다.

절대 규칙:
- JSON 외 텍스트(설명, 문장, 주석, 마크다운, ```) 금지
- 키 누락, 추가, 이름 변경 금지
- 값은 JSON 값만 사용 (true/false/null/number/string/object/array)
- 불확실하거나 이미지에 없으면 null
- 추측, 법적 판단, 요약, 해석 금지
""".strip()

    user_prompt = """
아래 이미지에서 문서에 '명시적으로 보이는' 내용만 근거로 추출하라.

반환은 반드시 아래 JSON 스켈레톤을 그대로 사용하라.
JSON 외의 어떤 문자도 출력하지 마라.

{
  "rules": {
    "deposit_over_market": {"flag": false, "evidence": []},
    "deposit_near_sale_price": {"flag": false, "evidence": []},

    "owner_mismatch": {"flag": false, "evidence": []},
    "proxy_without_power": {"flag": false, "evidence": []},
    "partial_owner_contract": {"flag": false, "evidence": []},

    "waive_rights": {"flag": false, "evidence": []},
    "allow_mortgage": {"flag": false, "evidence": []},
    "no_liability_return_delay": {"flag": false, "evidence": []},

    "move_in_restricted": {"flag": false, "evidence": []},
    "delayed_move_in": {"flag": false, "evidence": []},

    "short_term_contract": {"flag": false, "evidence": []},
    "one_sided_termination": {"flag": false, "evidence": []},
    "manual_modification": {"flag": false, "evidence": []},

    "broker_info_missing": {"flag": false, "evidence": []},
    "no_explanation_doc": {"flag": false, "evidence": []}
  },
  "extracted": {
    "deposit": {"value": null, "confidence": 0.0},
    "monthly_rent": {"value": null, "confidence": 0.0},
    "term_months": {"value": null, "confidence": 0.0},
    "landlord_name": {"text": null, "confidence": 0.0},
    "tenant_name": {"text": null, "confidence": 0.0},
    "address": {"text": null, "confidence": 0.0},
    "contract_date": {"text": null, "confidence": 0.0},
    "special_terms_raw": {"text": null, "confidence": 0.0}
  },
  "meta": {"uncertain_fields": []}
}

추가 규칙:
- evidence는 해당 판단을 직접 뒷받침하는 '문서의 문구 일부'를 짧게 넣어라(최대 3개).
- 금액은 숫자만 (원/만원/콤마/문자 제거)
- term_months는 개월 수 숫자만
- confidence는 0.0~1.0
- 확신이 낮은 필드는 meta.uncertain_fields에 필드 경로 문자열로 추가
- 시장가/매매가 대비 과다(예: deposit_over_market, deposit_near_sale_price)는
  계약서만으로 판단 불가능하면 flag=false로 두고 evidence도 비워라.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_vision_content(user_prompt, images_b64)},
        ],
        temperature=0,
        max_tokens=1400,
    )

    raw = (resp.choices[0].message.content or "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Vision JSON 파싱 실패(CONTRACT)", extra={"raw": raw})
        raise ValueError("Vision AI JSON 파싱 실패(CONTRACT)")


# -------------------------------------------------
# 외부에서 호출할 함수: analyze_contract(images_b64)
# -------------------------------------------------
def analyze_contract(images_b64: List[str]) -> Tuple[int, List[str], str, Dict[str, Any]]:
    db = SessionLocal()
    try:
        policy = get_active_policy(db)
        if not policy:
            raise ValueError("활성 정책이 없습니다(CONTRACT).")

        rules = get_active_rules(db, policy.id)

        parsed_data = parse_contract_info(images_b64)
        score, reasons = calculate_contract_risk_score(parsed_data, rules)

        return score, reasons, policy.version, parsed_data
    finally:
        db.close()
