from sqlalchemy import Column, Integer, String, ForeignKey, Date, Text
from db import Base


class RegistryRiskPolicy(Base):
    __tablename__ = "registry_risk_policy"

    id = Column(Integer, primary_key=True)
    version = Column(String(20), nullable=False)
    description = Column(String(500))
    is_active = Column(Integer, nullable=False)
    created_at = Column(Date)


class RegistryRiskRule(Base):
    __tablename__ = "registry_risk_rule"

    id = Column(Integer, primary_key=True)

    policy_id = Column(Integer, ForeignKey("registry_risk_policy.id"))
    category = Column(String(50), nullable=False)
    rule_key = Column(String(100), nullable=False)

    description = Column(String(500))
    score = Column(Integer, nullable=False)
    is_active = Column(Integer, nullable=False)

    created_at = Column(Date)


class RegistryRiskPrompt(Base):
    __tablename__ = "registry_risk_prompt"

    id = Column(Integer, primary_key=True)
    policy_id = Column(Integer, ForeignKey("registry_risk_policy.id"))
    prompt_template = Column(Text, nullable=False)
    created_at = Column(Date)
