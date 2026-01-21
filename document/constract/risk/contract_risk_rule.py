from sqlalchemy.orm import Session
from document.constract.risk.contract_risk import ContractRiskRule

def get_active_rules(db: Session, policy_id: int) -> list[ContractRiskRule]:
    rules = (
        db.query(ContractRiskRule)
        .filter(
            ContractRiskRule.policy_id == policy_id,
            ContractRiskRule.is_active == 1
        )
        .all()
    )

    if not rules:
        raise RuntimeError(f"정책 ID {policy_id}에 활성 룰이 없습니다.")

    return rules
