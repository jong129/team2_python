import base64
import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from tool import logger
from db import SessionLocal
from risk.contract_risk_policy import get_active_policy
from risk.contract_risk_rule import get_active_rules

client = OpenAI()


# -------------------------------------------------
# 이미지 → base64
# -------------------------------------------------
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -------------------------------------------------
# Vision OCR + 구조화
# -------------------------------------------------
def parsing_document_info(image_path: str) -> dict:
    img_base64 = encode_image(image_path)

    # -------------------------------
    # System 프롬프트 (형식 강제)
    # -------------------------------
    system_prompt = """
너는 등기 및 권리관계 문서 이미지에서 정보를 추출해
오직 JSON 객체 1개만 출력하는 OCR 파서다.

절대 규칙:
- JSON 외 텍스트(설명, 문장, 주석, 마크다운, ```) 금지
- 키 누락, 추가, 이름 변경 금지
- 값은 JSON 값만 사용 (true/false/null/number/string/object/array)
- 불확실하거나 이미지에 없으면 null
- 추측, 법적 판단, 요약, 해석 금지
"""

    # -------------------------------
    # User 프롬프트 (유효 JSON 스켈레톤)
    # -------------------------------
    user_prompt = """
아래 이미지에서 명시적으로 확인되는 사실만 추출하라.

반환은 반드시 아래 JSON 스켈레톤을 그대로 사용하라.
JSON 외의 어떤 문자도 출력하지 마라.

{
  "임차권": {
    "exists": false,
    "is_prior": null,
    "deposit": null
  },
  "압류": {
    "exists": false,
    "type": null,
    "count": null
  },
  "가압류": {
    "exists": false,
    "amount": null
  },
  "근저당": {
    "exists": false,
    "max_amount": null
  },
  "신탁": {
    "exists": false
  },
  "meta": {
    "uncertain_fields": []
  }
}

추가 규칙:
- exists / is_prior 는 true 또는 false만 사용
- 금액, 개수는 숫자만 사용 (원, 콤마, 문자 금지)
- type 은 "국세" | "지방세" | "기타" 중 하나만 사용, 불명확하면 null
- 확신이 낮은 필드는 meta.uncertain_fields 에 필드 경로 문자열로 추가
"""

    # -------------------------------
    # Vision 호출 (JSON mode)
    # -------------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt.strip()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=1200
    )

    raw = response.choices[0].message.content.strip()

    # -------------------------------
    # 1차 파싱
    # -------------------------------
    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        logger.error("Vision JSON 파싱 실패 (1차)", extra={"raw": raw})

        # -------------------------------
        # 리페어 1회 재시도
        # -------------------------------
        repair_prompt = f"""
아래 출력은 JSON 형식 위반이다.
오직 유효한 JSON 객체 1개로만 고쳐라.

규칙:
- JSON 외 텍스트 금지
- 키는 스켈레톤과 동일
- 불확실하면 null

원본 출력:
{raw}
"""

        repair_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "너는 JSON 수리기다. JSON만 출력하라."},
                {"role": "user", "content": repair_prompt.strip()},
            ],
            temperature=0,
            max_tokens=800
        )

        repaired_raw = repair_resp.choices[0].message.content.strip()

        try:
            return json.loads(repaired_raw)
        except json.JSONDecodeError:
            logger.error(
                "Vision JSON 파싱 실패 (리페어 후)",
                extra={"raw": raw, "repaired": repaired_raw}
            )
            raise ValueError("Vision AI JSON 파싱 실패")


# -------------------------------------------------
# 위험 점수 계산 (DB 룰 기반)
# -------------------------------------------------
def calculate_risk_score(parsed_data: dict, rules):
    total_score = 0
    reasons = []

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

    return total_score, reasons


# -------------------------------------------------
# AI 설명 생성 (설명만!)
# -------------------------------------------------
def generate_ai_explanation(score: int, reasons: list, policy_version: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
너는 전세계약 위험도 결과를 설명하는 AI다.
새로운 판단이나 점수 계산을 절대 하지 마라.

[정책 버전]
{policy_version}

[위험 점수]
{score}점

[판단 근거]
{chr(10).join(f"- {r}" for r in reasons)}
"""

    resp = llm.invoke([SystemMessage(content=prompt)])
    return resp.content


# -------------------------------------------------
# 최종 진입 함수 
# -------------------------------------------------
def analyze_document(image_path: str) -> dict:
    db = SessionLocal()
    print("① DB 세션 생성 완료")

    try:
        print("② 정책 조회 시작")
        policy = get_active_policy(db)
        print("② 정책 조회 결과:", policy)

        if not policy:
            raise ValueError("활성 정책이 없습니다.")

        print("③ 룰 조회 시작")
        rules = get_active_rules(db, policy.id)
        print("③ 룰 개수:", len(rules))

        print("④ Vision 파싱 시작")
        parsed_data = parsing_document_info(image_path)
        print("④ Vision 파싱 결과:", parsed_data)

        print("⑤ 점수 계산 시작")
        score, reasons = calculate_risk_score(parsed_data, rules)
        print("⑤ 점수:", score, "사유:", reasons)

        print("⑥ AI 설명 생성 시작")
        explanation = generate_ai_explanation(score, reasons, policy.version)
        print("⑥ AI 설명 완료")

        return {
            "policy_version": policy.version,
            "risk_score": score,
            "reasons": reasons,
            "ai_explanation": explanation,
            "parsed_data": parsed_data
        }

    except Exception as e:
        print("💥 EXCEPTION TYPE:", type(e))
        print("💥 EXCEPTION MSG:", e)
        logger.error("문서 분석 실패", exc_info=True)
        raise

    finally:
        print("⑦ DB 세션 종료")
        db.close()
