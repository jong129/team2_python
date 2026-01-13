CREATE TABLE registry_risk_policy (
    id NUMBER
        GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    version VARCHAR2(20) NOT NULL,
    description VARCHAR2(500),

    is_active NUMBER(1) DEFAULT 0 NOT NULL,

    created_at DATE DEFAULT SYSDATE
);

COMMENT ON TABLE registry_risk_policy IS '전세계약 위험도 정책(버전 단위)';
COMMENT ON COLUMN registry_risk_policy.id IS '정책 ID (자동 증가)';
COMMENT ON COLUMN registry_risk_policy.version IS '정책 버전';
COMMENT ON COLUMN registry_risk_policy.description IS '정책 설명';
COMMENT ON COLUMN registry_risk_policy.is_active IS '현재 적용 여부 (1=활성, 0=비활성)';
COMMENT ON COLUMN registry_risk_policy.created_at IS '생성 시각';

INSERT INTO registry_risk_policy (
    version,
    description,
    is_active,
    created_at
) VALUES (
    'v1.0',
    '초기 전세계약 위험도 산정 정책
- 등기부 등본에 명시된 권리관계만 사용
- 점수 계산은 서버 로직에서 수행
- AI는 설명만 담당',
    1,
    SYSDATE
);
COMMIT;

CREATE TABLE registry_risk_rule (
    id NUMBER
        GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    policy_id NUMBER NOT NULL,

    category VARCHAR2(50) NOT NULL,
    rule_key VARCHAR2(100) NOT NULL,

    description VARCHAR2(500),
    score NUMBER NOT NULL,

    is_active NUMBER(1) DEFAULT 1 NOT NULL,

    created_at DATE DEFAULT SYSDATE,

    CONSTRAINT fk_registry_rule_policy
        FOREIGN KEY (policy_id)
        REFERENCES registry_risk_policy(id)
);

COMMENT ON TABLE registry_risk_rule IS '전세계약 위험도 산정 개별 룰';
COMMENT ON COLUMN registry_risk_rule.id IS '룰 ID (자동 증가)';
COMMENT ON COLUMN registry_risk_rule.policy_id IS '정책 ID';
COMMENT ON COLUMN registry_risk_rule.category IS '위험 분류 (임차권, 압류, 가압류, 근저당, 신탁)';
COMMENT ON COLUMN registry_risk_rule.rule_key IS '룰 식별 키 (exists, is_prior, tax 등)';
COMMENT ON COLUMN registry_risk_rule.description IS '룰 설명';
COMMENT ON COLUMN registry_risk_rule.score IS '부여 점수';
COMMENT ON COLUMN registry_risk_rule.is_active IS '룰 활성 여부 (1=활성)';
COMMENT ON COLUMN registry_risk_rule.created_at IS '생성 시각';

INSERT INTO registry_risk_rule (
    policy_id, category, rule_key, description, score, is_active, created_at
) VALUES (
    1,
    '임차권',
    'exists',
    '등기부에 주택임차권 등기가 존재함',
    30,
    1,
    SYSDATE
);

INSERT INTO registry_risk_rule (
    policy_id, category, rule_key, description, score, is_active, created_at
) VALUES (
    1,
    '임차권',
    'is_prior',
    '선순위 주택임차권 존재',
    20,
    1,
    SYSDATE
);

INSERT INTO registry_risk_rule (
    policy_id, category, rule_key, description, score, is_active, created_at
) VALUES (
    1,
    '압류',
    'tax',
    '국세 또는 지방세 압류 존재',
    25,
    1,
    SYSDATE
);

INSERT INTO registry_risk_rule (
    policy_id, category, rule_key, description, score, is_active, created_at
) VALUES (
    1,
    '압류',
    'other',
    '기타 채권 압류 존재',
    15,
    1,
    SYSDATE
);

INSERT INTO registry_risk_rule (
    policy_id, category, rule_key, description, score, is_active, created_at
) VALUES (
    1,
    '가압류',
    'exists',
    '가압류 등기 존재',
    15,
    1,
    SYSDATE
);

INSERT INTO registry_risk_rule (
    policy_id, category, rule_key, description, score, is_active, created_at
) VALUES (
    1,
    '근저당',
    'exists',
    '근저당권 설정 존재',
    20,
    1,
    SYSDATE
);

INSERT INTO registry_risk_rule (
    policy_id, category, rule_key, description, score, is_active, created_at
) VALUES (
    1,
    '신탁',
    'exists',
    '신탁 등기 존재 (실소유자 불일치 위험)',
    50,
    1,
    SYSDATE
);


COMMIT;

CREATE TABLE registry_risk_prompt (
    id NUMBER
        GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    policy_id NUMBER NOT NULL,

    prompt_template CLOB NOT NULL,

    created_at DATE DEFAULT SYSDATE,

    CONSTRAINT fk_registry_prompt_policy
        FOREIGN KEY (policy_id)
        REFERENCES contract_risk_policy(id)
);

COMMENT ON TABLE registry_risk_prompt IS '전세계약 위험도 설명용 AI 프롬프트';
COMMENT ON COLUMN registry_risk_prompt.id IS '프롬프트 ID (자동 증가)';
COMMENT ON COLUMN registry_risk_prompt.policy_id IS '정책 ID';
COMMENT ON COLUMN registry_risk_prompt.prompt_template IS 'AI 설명 프롬프트 템플릿';
COMMENT ON COLUMN registry_risk_prompt.created_at IS '생성 시각';

INSERT INTO registry_risk_prompt (
    policy_id,
    prompt_template,
    created_at
) VALUES (
    1,
    '너는 전세계약 위험도 분석 결과를 설명하는 AI다.

[역할 제한]
- 새로운 점수 계산을 하지 마라
- 법적 판단이나 추론을 하지 마라
- 제공된 결과를 해석하여 설명만 수행하라

[출력 형식]
위험도: {{RISK_PERCENT}}%

근거:
{{REASONS}}

[설명 규칙]
- 각 근거는 등기부에 실제 존재하는 항목만 사용
- 과도한 추측이나 단정적 표현을 사용하지 마라
- 사실 기반으로 간결하게 설명하라

[계산 결과 데이터]
{{RESULT_JSON}}',
    SYSDATE
);
COMMIT;
