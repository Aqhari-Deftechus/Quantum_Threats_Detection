from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class FusionSignal:
    module_name: str
    module_version: str
    contract_version: str
    decision: str
    reason_code: str
    timestamp: str


def demo_fusion_signal() -> FusionSignal:
    return FusionSignal(
        module_name="fusion_demo",
        module_version="0.1.0",
        contract_version="0.1.0",
        decision="ALLOW",
        reason_code="DEMO_RULE",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
