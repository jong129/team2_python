CREATE TABLE contract_risk_policy (
    id NUMBER
        GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    version VARCHAR2(20) NOT NULL,
    description VARCHAR2(500),

    is_active NUMBER(1) DEFAULT 0 NOT NULL,

    created_at DATE DEFAULT SYSDATE
);

COMMENT ON TABLE contract_risk_policy IS '전세계약 위험도 정책(버전 단위)';
COMMENT ON COLUMN contract_risk_policy.id IS '정책 ID (자동 증가)';
COMMENT ON COLUMN contract_risk_policy.version IS '정책 버전';
COMMENT ON COLUMN contract_risk_policy.description IS '정책 설명';
COMMENT ON COLUMN contract_risk_policy.is_active IS '현재 적용 여부 (1=활성, 0=비활성)';
COMMENT ON COLUMN contract_risk_policy.created_at IS '생성 시각';

INSERT INTO contract_risk_policy (
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

CREATE TABLE contract_risk_rule (
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

    CONSTRAINT fk_cr_rule_policy
        FOREIGN KEY (policy_id)
        REFERENCES contract_risk_policy(id)
);

COMMENT ON TABLE contract_risk_rule IS '전세계약 위험도 산정 개별 룰';
COMMENT ON COLUMN contract_risk_rule.id IS '룰 ID (자동 증가)';
COMMENT ON COLUMN contract_risk_rule.policy_id IS '정책 ID';
COMMENT ON COLUMN contract_risk_rule.category IS '위험 분류 (임차권, 압류, 가압류, 근저당, 신탁)';
COMMENT ON COLUMN contract_risk_rule.rule_key IS '룰 식별 키 (exists, is_prior, tax 등)';
COMMENT ON COLUMN contract_risk_rule.description IS '룰 설명';
COMMENT ON COLUMN contract_risk_rule.score IS '부여 점수';
COMMENT ON COLUMN contract_risk_rule.is_active IS '룰 활성 여부 (1=활성)';
COMMENT ON COLUMN contract_risk_rule.created_at IS '생성 시각';

-- 보증금 관련
INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'deposit_over_market',
  '보증금이 시세 대비 과도하게 높음', 30, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'deposit_near_sale_price',
  '보증금이 매매가에 근접함', 40, 1, SYSDATE
);

-- 임대인 / 대리인
INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'owner_mismatch',
  '등기상 소유자와 계약자가 일치하지 않음', 50, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'proxy_without_power',
  '대리 계약이나 위임장 확인 불가', 40, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'partial_owner_contract',
  '공동 소유 부동산 중 일부 소유자만 계약', 35, 1, SYSDATE
);

-- 특약 위험
INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'waive_rights',
  '임차인의 권리를 포기하는 특약 존재', 45, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'allow_mortgage',
  '임대인이 추가 근저당 설정 가능 특약', 40, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'no_liability_return_delay',
  '보증금 반환 지연에 대한 책임 면제 조항', 50, 1, SYSDATE
);

-- 입주 / 기간
INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'move_in_restricted',
  '입주 조건 또는 입주 제한 특약 존재', 30, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'delayed_move_in',
  '입주 가능일이 계약일보다 지연됨', 25, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'short_term_contract',
  '계약 기간이 비정상적으로 짧음', 20, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'one_sided_termination',
  '일방적 계약 해지 조항 존재', 45, 1, SYSDATE
);

-- 문서 / 중개
INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'manual_modification',
  '수기 수정 또는 임의 기재 흔적 존재', 20, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'broker_info_missing',
  '중개사 정보 누락', 35, 1, SYSDATE
);

INSERT INTO CONTRACT_RISK_RULE (
  POLICY_ID, CATEGORY, RULE_KEY, DESCRIPTION, SCORE, IS_ACTIVE, CREATED_AT
) VALUES (
  1, 'CONTRACT', 'no_explanation_doc',
  '중요사항 설명서 미첨부', 30, 1, SYSDATE
);

COMMIT;
