from sqlalchemy.orm import Session
from document.registry.risk.registry_risk import RegistryRiskPolicy


def get_active_policy(db: Session) -> RegistryRiskPolicy:
    policy = (
        db.query(RegistryRiskPolicy)
        .filter(RegistryRiskPolicy.is_active == 1)
        .order_by(RegistryRiskPolicy.created_at.desc())
        .first()
    )

    if not policy:
        raise RuntimeError("활성화된 위험도 정책이 존재하지 않습니다.")

    return policy
