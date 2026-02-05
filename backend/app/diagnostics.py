from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RollingMetric:
    total_seconds: float = 0.0
    count: int = 0

    def update(self, duration_seconds: float) -> None:
        self.total_seconds += duration_seconds
        self.count += 1

    def average_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return (self.total_seconds / self.count) * 1000.0

    def should_log(self, every_n: int) -> bool:
        return every_n > 0 and self.count % every_n == 0
