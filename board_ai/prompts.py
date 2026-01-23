def build_summary_prompt(db_prompt: str, title: str, content: str) -> str:
    return f"""{db_prompt}

[제목]
{title}

[본문]
{content}

요구사항:
- 한국어로 작성
- 핵심만 3~5줄로 요약
- 과장/추측/투자조언 금지
- 사실만 간결하게
""".strip()

def build_sentiment_prompt(db_prompt: str, title: str, content: str) -> str:
    return f"""{db_prompt}

[제목]
{title}

[본문]
{content}

요구사항:
- 부동산 시장 관점에서 '호재/악재/혼합' 중 하나로 판단
- 관점은 부동산 직종인이 아닌 집을 구하려는 일반 소비자 관점
- 본문이 한 줄 이하면 '판단 불가'로 응답
- 출력 형식:
  1) 결론: (호재/악재/혼합)
  2) 근거: 2~3줄
- 과장/추측/투자조언 금지
""".strip()

def build_write_prompt(db_prompt: str, title: str, content: str) -> str:
    return f"""{db_prompt}

[사용자 입력 제목(있으면 참고)]
{title}

[사용자 입력 본문(초안)]
{content}

요구사항:
- 한국어로 작성
- 사용자 입력(제목/본문)에 포함된 사실만 사용 (없는 내용은 만들지 말 것)
- 인터넷 커뮤니티에 쓸법한 자연스러운 문체
- 과장/추측/투자조언/홍보문구 금지
- 게시글 초안 형태로 자연스럽게 문장을 다듬기
- 결과는 '완성된 본문'만 출력
- 길이는 너무 길지 않게 8~20줄 내에서 상황에 맞게
""".strip()

def build_moderate_image_prompt(db_prompt: str, filename: str, content_type: str) -> str:
    return f"""{db_prompt}

[파일명]
{filename}

[Content-Type]
{content_type}

작업:
- 이미지를 보고 게시판 업로드를 허용할지 판단한다.
- 기준 카테고리:
  - AD/COMMERCIAL: 광고/상업 홍보(전단, 가격표, 연락처/URL/QR, 상호/로고 과도 등)
  - SEXUAL: 선정적/노출/성적 암시
  - VIOLENCE: 폭력/잔혹/혐오감 유발
  - HATE: 혐오 표현/상징
  - OTHER: 기타 부적절(사칭/불법/불쾌한 장면 등)
- 애매하면 allowed=false(보수적)로 판단한다.
- score는 판단 확신도(0.0~1.0).
- 한국 영상물 등급 심의 기준 15세 이하의 수준까지는 인정(안기, 손잡기 등 정상적 신체 접촉은 허용).
출력 규칙(절대 준수):
- 반드시 JSON 한 줄만 출력한다. 다른 텍스트 금지.
- 키 이름 고정:
  {{"allowed":true|false,"reason_code":"AD|COMMERCIAL|SEXUAL|VIOLENCE|HATE|OTHER","reason_text":"한글 1줄","score":0.0}}
""".strip()
