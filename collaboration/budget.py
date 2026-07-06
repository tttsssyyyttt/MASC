"""Budget manager for SEA intervention."""

from __future__ import annotations


class BudgetManager:
    """Manages intervention budget: total budget, cooldown, usage tracking."""

    def __init__(self, total_budget: int = 100, cooldown: int = 5):
        self.total_budget = total_budget
        self.cooldown = cooldown
        self.used = 0
        self.last_intervention_step = -cooldown  # allow immediate intervention

    def can_intervene(self, step: int) -> bool:
        """Check if intervention is allowed given budget and cooldown."""
        if self.used >= self.total_budget:
            return False
        if step - self.last_intervention_step < self.cooldown:
            return False
        return True

    def record_intervention(self, step: int):
        """Record that an intervention occurred."""
        self.used += 1
        self.last_intervention_step = step

    def reset(self):
        """Reset for a new episode."""
        self.used = 0
        self.last_intervention_step = -self.cooldown
