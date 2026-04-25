"""OpenEnv-style training environment for zombie API detection and mitigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple
import random

try:
    # Preferred path for `openenv-core==0.2.3+`.
    from openenv.core import Environment  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Compatibility fallback for older OpenEnv layouts.
        from openenv import Environment  # type: ignore
    except Exception:
        class Environment:  # fallback for local/Colab execution
            def reset(self):
                raise NotImplementedError

            def step(self, action):
                raise NotImplementedError

            def state(self):
                raise NotImplementedError

from env.reward_engine import RewardEngine
from env.state_generator import StateGenerator
from simulator.api_simulator import APISimulator


VALID_ACTIONS = {
    "scan_api",
    "classify_api",
    "run_security_test",
    "block_api",
    "ignore_api",
    "escalate_api",
}


@dataclass
class EpisodeStats:
    vulnerabilities_detected: int = 0
    correct_classifications: int = 0
    incorrect_classifications: int = 0
    blocked_zombies: int = 0
    blocked_active: int = 0


class ZombieShieldEnv(Environment):
    """RL environment for discovering and mitigating zombie APIs."""

    def __init__(
        self,
        min_apis: int = 10,
        max_apis: int = 50,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        max_actions_per_api: int = 10,
    ):
        self.min_apis = min_apis
        self.max_apis = max_apis
        self.seed = seed
        self._rng = random.Random(seed)
        self.max_actions_per_api = max_actions_per_api

        self.simulator = APISimulator(seed=seed)
        self.state_generator = StateGenerator()
        self.reward_engine = RewardEngine()

        self.configured_max_steps = max_steps
        self.max_steps = max_steps
        self.step_count = 0
        self.visible_api_ids: Set[str] = set()
        self.known_metadata: Dict[str, Dict] = {}
        self.tested_results: Dict[str, Dict] = {}
        self.predicted_labels: Dict[str, str] = {}
        self.processed_api_ids: Set[str] = set()
        self.last_actions: Set[Tuple[str, str]] = set()
        self.api_action_counts: Dict[str, int] = {}
        self.stats = EpisodeStats()

    def reset(self) -> Dict:
        num_apis = self._rng.randint(self.min_apis, self.max_apis)
        self.simulator.reset(num_apis=num_apis)

        self.step_count = 0
        self.visible_api_ids = set(self.simulator.discoverable_api_ids())
        self.known_metadata = {api_id: {"api_id": api_id, "observed": False} for api_id in self.visible_api_ids}
        self.tested_results = {}
        self.predicted_labels = {}
        self.processed_api_ids = set()
        self.last_actions = set()
        self.api_action_counts = {}
        self.stats = EpisodeStats()

        if self.configured_max_steps is None:
            self.max_steps = max(30, num_apis * 4)
        else:
            self.max_steps = self.configured_max_steps
        self.reward_engine.reset()

        return self.state()

    def state(self) -> Dict:
        return self.state_generator.build(
            simulator=self.simulator,
            known_metadata=self.known_metadata,
            tested_results=self.tested_results,
            visible_api_ids=self.visible_api_ids,
            processed_api_ids=self.processed_api_ids,
            step_count=self.step_count,
            max_steps=self.max_steps or 1,
        )

    def step(self, action):
        parsed = self._parse_action(action)
        self.step_count += 1

        reward = 0.0
        info: Dict = {"action": parsed, "status": "ok"}

        if parsed is None:
            reward += self.reward_engine.action_penalty(invalid=True)
            info["status"] = "invalid_action"
            done = self._is_done()
            if done:
                reward += self._terminal_adjustments(info)
            return self.state(), reward, done, info

        action_type = parsed["type"]
        api_id = parsed.get("api_id")

        repeated = (action_type, str(api_id)) in self.last_actions
        reward += self.reward_engine.action_penalty(repeated=repeated)
        self.last_actions.add((action_type, str(api_id)))

        if action_type in {"scan_api", "run_security_test", "classify_api", "block_api", "ignore_api", "escalate_api"}:
            if api_id not in self.visible_api_ids:
                reward += self.reward_engine.action_penalty(invalid=True)
                info["status"] = "api_not_visible"
                done = self._is_done()
                if done:
                    reward += self._terminal_adjustments(info)
                return self.state(), reward, done, info
            self.api_action_counts[api_id] = self.api_action_counts.get(api_id, 0) + 1
            if self.api_action_counts[api_id] > self.max_actions_per_api:
                reward -= 4.0
                self.processed_api_ids.add(api_id)
                info["status"] = "api_action_budget_exceeded"
                info["api_action_count"] = self.api_action_counts[api_id]
                done = self._is_done()
                if done:
                    reward += self._terminal_adjustments(info)
                return self.state(), reward, done, info

        if action_type == "scan_api":
            ok, result = self.simulator.scan_api(api_id)
            if ok:
                self.known_metadata[api_id] = {**result, "observed": True}
                discovered = result.get("new_discovery")
                if discovered is not None and discovered not in self.visible_api_ids:
                    self.visible_api_ids.add(discovered)
                    self.known_metadata[discovered] = {"api_id": discovered, "observed": False}
                    info["new_discovery"] = discovered
                reward += 1.0
            else:
                reward -= 1.0
                info["scan_error"] = result.get("error", "unknown")

        elif action_type == "classify_api":
            label = parsed.get("label")
            if label not in {"ACTIVE", "ZOMBIE"}:
                reward += self.reward_engine.action_penalty(invalid=True)
                info["status"] = "invalid_label"
            else:
                api = self.simulator.get_api(api_id)
                assert api is not None
                true_label = "ZOMBIE" if api.is_zombie else "ACTIVE"
                self.predicted_labels[api_id] = label
                reward_delta = self.reward_engine.classify_reward(api_id, true_label, label)
                reward += reward_delta
                if reward_delta > 0:
                    self.stats.correct_classifications += 1
                else:
                    self.stats.incorrect_classifications += 1

        elif action_type == "run_security_test":
            ok, result = self.simulator.run_security_test(api_id)
            if not ok:
                reward -= 2.0
                info["test_error"] = result.get("error", "unknown")
            else:
                self.tested_results[api_id] = result
                detected = len([v for v in result.get("detected_vulnerabilities", []) if "false_signal" not in v])
                api = self.simulator.get_api(api_id)
                assert api is not None
                true_label = "ZOMBIE" if api.is_zombie else "ACTIVE"
                reward += self.reward_engine.vulnerability_reward(api_id, true_label, detected)
                self.stats.vulnerabilities_detected += detected

        elif action_type == "block_api":
            api = self.simulator.get_api(api_id)
            assert api is not None
            true_label = "ZOMBIE" if api.is_zombie else "ACTIVE"
            classified_first = api_id in self.predicted_labels
            confirmed = self.predicted_labels.get(api_id) == "ZOMBIE" or (
                api_id in self.tested_results and len(self.tested_results[api_id].get("detected_vulnerabilities", [])) > 0
            )
            reward += self.reward_engine.block_reward(api_id, true_label, confirmed, classified_first=classified_first)
            if not classified_first:
                reward -= 3.0
                info["classification_first_missing"] = True
            self.processed_api_ids.add(api_id)
            if api.is_zombie:
                self.stats.blocked_zombies += 1
            else:
                self.stats.blocked_active += 1

        elif action_type == "ignore_api":
            if api_id not in self.predicted_labels:
                reward -= 2.0
                info["classification_first_missing"] = True
            self.processed_api_ids.add(api_id)
            reward -= 0.5

        elif action_type == "escalate_api":
            if api_id not in self.predicted_labels:
                reward -= 2.0
                info["classification_first_missing"] = True
            self.processed_api_ids.add(api_id)
            # Escalation helps cautiously with uncertain APIs.
            if api_id in self.tested_results and self.tested_results[api_id].get("detected_vulnerabilities"):
                reward += 3.0
            else:
                reward += 0.5

        done = self._is_done()
        if done:
            reward += self._terminal_adjustments(info)

        info["step"] = self.step_count
        info["episode_stats"] = self._stats_payload()
        info["cumulative_processed"] = len(self.processed_api_ids)

        return self.state(), reward, done, info

    def _stats_payload(self) -> Dict:
        return {
            "vulnerabilities_detected": self.stats.vulnerabilities_detected,
            "correct_classifications": self.stats.correct_classifications,
            "incorrect_classifications": self.stats.incorrect_classifications,
            "blocked_zombies": self.stats.blocked_zombies,
            "blocked_active": self.stats.blocked_active,
        }

    def _terminal_adjustments(self, info: Dict) -> float:
        zombie_ids = [api.api_id for api in self.simulator.apis if api.is_zombie]
        accuracy = self._classification_accuracy()
        metrics = self._classification_metrics()
        terminal_reward = self.reward_engine.terminal_reward(
            zombie_api_ids=zombie_ids,
            predicted_labels=self.predicted_labels,
            step_count=self.step_count,
            max_steps=self.max_steps or 1,
            recall=metrics["recall"],
            f1=metrics["f1"],
            labeled_fraction=metrics["labeled_fraction"],
        )
        info["terminal_accuracy"] = accuracy
        info["terminal_precision"] = metrics["precision"]
        info["terminal_recall"] = metrics["recall"]
        info["terminal_f1"] = metrics["f1"]
        info["terminal_labeled_fraction"] = metrics["labeled_fraction"]
        info["terminal_tp"] = metrics["tp"]
        info["terminal_fp"] = metrics["fp"]
        info["terminal_tn"] = metrics["tn"]
        info["terminal_fn"] = metrics["fn"]
        info["terminal_vulnerabilities_detected"] = self.stats.vulnerabilities_detected
        return terminal_reward

    def _classification_accuracy(self) -> float:
        if not self.predicted_labels:
            return 0.0

        correct = 0
        for api_id, label in self.predicted_labels.items():
            api = self.simulator.get_api(api_id)
            if api is None:
                continue
            truth = "ZOMBIE" if api.is_zombie else "ACTIVE"
            if truth == label:
                correct += 1

        return round(correct / max(1, len(self.predicted_labels)), 4)

    def _classification_metrics(self) -> Dict:
        tp = fp = tn = fn = 0
        all_api_ids = [api.api_id for api in self.simulator.apis]
        labeled = 0

        for api in self.simulator.apis:
            pred = self.predicted_labels.get(api.api_id)
            if pred is not None:
                labeled += 1

            true_is_zombie = api.is_zombie
            pred_is_zombie = pred == "ZOMBIE"

            if true_is_zombie and pred_is_zombie:
                tp += 1
            elif true_is_zombie and not pred_is_zombie:
                fn += 1
            elif (not true_is_zombie) and pred_is_zombie:
                fp += 1
            else:
                tn += 1

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-9, precision + recall)

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "labeled_fraction": round(labeled / max(1, len(all_api_ids)), 4),
        }

    def _is_done(self) -> bool:
        if self.step_count >= (self.max_steps or 1):
            return True
        return len(self.processed_api_ids) >= len(self.simulator.apis)

    def _parse_action(self, action) -> Optional[Dict]:
        if isinstance(action, dict):
            action_type = action.get("type")
            if action_type not in VALID_ACTIONS:
                return None
            return action

        if not isinstance(action, str):
            return None

        action = action.strip()
        if "(" not in action or not action.endswith(")"):
            return None

        action_type, args_part = action.split("(", 1)
        action_type = action_type.strip()
        args_part = args_part[:-1]

        if action_type not in VALID_ACTIONS:
            return None

        args = [chunk.strip() for chunk in args_part.split(",") if chunk.strip()]

        if action_type == "classify_api":
            if len(args) != 2:
                return None
            return {"type": action_type, "api_id": args[0], "label": args[1].upper()}

        if action_type in {"scan_api", "run_security_test", "block_api", "ignore_api", "escalate_api"}:
            if len(args) != 1:
                return None
            return {"type": action_type, "api_id": args[0]}

        return None
