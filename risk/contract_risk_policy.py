from sqlalchemy.orm import Session
from risk.contract_risk import ContractRiskPolicy


def get_active_policy(db: Session) -> ContractRiskPolicy:
    policy = (
        db.query(ContractRiskPolicy)
        .filter(ContractRiskPolicy.is_active == 1)
        .order_by(ContractRiskPolicy.created_at.desc())
        .first()
    )

    if not policy:
        raise RuntimeError("활성화된 위험도 정책이 존재하지 않습니다.")

    return policy
