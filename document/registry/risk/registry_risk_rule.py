from sqlalchemy.orm import Session
from document.registry.risk.registry_risk import RegistryRiskRule

def get_active_rules(db: Session, policy_id: int) -> list[RegistryRiskRule]:
    rules = (
        db.query(RegistryRiskRule)
        .filter(
            RegistryRiskRule.policy_id == policy_id,
            RegistryRiskRule.is_active == 1
        )
        .all()
    )

    if not rules:
        raise RuntimeError(f"정책 ID {policy_id}에 활성 룰이 없습니다.")

    return rules
