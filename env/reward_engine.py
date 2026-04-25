"""Reward shaping and anti-gaming logic for ZombieShieldEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class RewardTracker:
    classified: Set[str] = field(default_factory=set)
    vuln_rewarded: Set[str] = field(default_factory=set)
    blocked_rewarded: Set[str] = field(default_factory=set)
    excessive_action_count: int = 0


class RewardEngine:
    def __init__(self) -> None:
        self.tracker = RewardTracker()

    def reset(self) -> None:
        self.tracker = RewardTracker()

    def action_penalty(self, repeated: bool = False, invalid: bool = False) -> float:
        reward = 0.0
        if repeated:
            self.tracker.excessive_action_count += 1
            reward -= 1.0
        if invalid:
            self.tracker.excessive_action_count += 1
            reward -= 2.0
        return reward

    def classify_reward(self, api_id: str, true_label: str, predicted_label: str) -> float:
        if api_id in self.tracker.classified:
            return -1.0

        self.tracker.classified.add(api_id)

        if predicted_label == "ZOMBIE" and true_label == "ZOMBIE":
            return 10.0
        if predicted_label == "ZOMBIE" and true_label == "ACTIVE":
            return -5.0
        if predicted_label == "ACTIVE" and true_label == "ZOMBIE":
            return -15.0
        return 2.0

    def vulnerability_reward(self, api_id: str, true_label: str, detected_count: int) -> float:
        if api_id in self.tracker.vuln_rewarded:
            return -0.5
        if detected_count <= 0:
            return -1.5

        self.tracker.vuln_rewarded.add(api_id)
        if true_label == "ZOMBIE":
            return 8.0
        return -1.0

    def block_reward(self, api_id: str, true_label: str, confirmed: bool, classified_first: bool) -> float:
        if api_id in self.tracker.blocked_rewarded:
            return -1.0

        self.tracker.blocked_rewarded.add(api_id)

        if true_label == "ZOMBIE" and confirmed and classified_first:
            return 15.0
        if true_label == "ZOMBIE" and confirmed and not classified_first:
            return 6.0
        if true_label == "ZOMBIE" and not confirmed and classified_first:
            return 2.0
        if true_label == "ZOMBIE" and not confirmed and not classified_first:
            return -2.0
        return -10.0 if classified_first else -12.0

    def terminal_reward(
        self,
        zombie_api_ids: List[str],
        predicted_labels: Dict[str, str],
        step_count: int,
        max_steps: int,
        recall: float = 0.0,
        labeled_fraction: float = 0.0,
    ) -> float:
        reward = 0.0

        for api_id in zombie_api_ids:
            if predicted_labels.get(api_id) != "ZOMBIE":
                reward -= 15.0

        efficiency = max(0.0, (max_steps - step_count) / max_steps)
        reward += efficiency * 3.0

        # Anti-gaming: discourage policies that avoid labeling/classification actions.
        if labeled_fraction < 0.30:
            reward -= (0.30 - labeled_fraction) * 30.0

        # Anti-gaming: discourage collapsing recall by over-conservative behavior.
        if recall < 0.20:
            reward -= (0.20 - recall) * 25.0

        # Penalize random/excessive interaction patterns.
        reward -= min(5.0, self.tracker.excessive_action_count * 0.2)
        return reward
