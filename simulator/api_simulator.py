"""Core API ecosystem simulator for ZombieShieldEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional, Tuple


AUTH_TYPES = ["none", "api_key", "oauth2", "jwt", "mTLS"]
VERSIONS = ["v1", "v2", "v3", "legacy"]


@dataclass
class APIAsset:
    api_id: str
    endpoint: str
    version: str
    auth_type: str
    traffic_frequency: float
    last_updated: str
    is_zombie: bool
    vulnerabilities: List[str] = field(default_factory=list)
    hidden: bool = False


class APISimulator:
    """Simulates a partially-observable API ecosystem with noisy feedback."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self.seed = seed
        self.apis: List[APIAsset] = []
        self.request_logs: List[Dict] = []

    def reset(self, num_apis: int) -> None:
        self.apis = self._generate_apis(num_apis)
        self.request_logs = []

    def _generate_apis(self, num_apis: int) -> List[APIAsset]:
        apis: List[APIAsset] = []
        now = datetime.utcnow()

        for idx in range(num_apis):
            version = self._rng.choice(VERSIONS)
            auth_type = self._rng.choices(AUTH_TYPES, weights=[0.2, 0.3, 0.25, 0.2, 0.05])[0]
            days_ago = self._rng.randint(0, 1200)
            last_updated = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            traffic = max(0.01, self._rng.random() * 100)
            hidden = self._rng.random() < 0.25

            zombie_score = 0
            zombie_score += 2 if version in {"legacy", "v1"} else 0
            zombie_score += 2 if auth_type == "none" else 0
            zombie_score += 2 if days_ago > 450 else 0
            zombie_score += 1 if traffic < 4 else 0
            zombie_noise = self._rng.randint(-1, 2)
            is_zombie = zombie_score + zombie_noise >= 4

            vulnerabilities: List[str] = []
            if is_zombie:
                candidates = [
                    "broken_authentication",
                    "deprecated_crypto",
                    "sensitive_data_exposure",
                    "mass_assignment",
                    "missing_rate_limiting",
                ]
                self._rng.shuffle(candidates)
                vul_count = self._rng.randint(1, 3)
                vulnerabilities = candidates[:vul_count]

            apis.append(
                APIAsset(
                    api_id=f"api_{idx}",
                    endpoint=f"/service/{idx}/{self._rng.choice(['users', 'payments', 'orders', 'internal'])}",
                    version=version,
                    auth_type=auth_type,
                    traffic_frequency=round(traffic, 2),
                    last_updated=last_updated,
                    is_zombie=is_zombie,
                    vulnerabilities=vulnerabilities,
                    hidden=hidden,
                )
            )

        return apis

    def discoverable_api_ids(self) -> List[str]:
        return [api.api_id for api in self.apis if not api.hidden]

    def maybe_discover_hidden_api(self) -> Optional[str]:
        hidden = [api for api in self.apis if api.hidden]
        if not hidden:
            return None
        if self._rng.random() > 0.30:
            return None
        target = self._rng.choice(hidden)
        target.hidden = False
        return target.api_id

    def get_api(self, api_id: str) -> Optional[APIAsset]:
        for api in self.apis:
            if api.api_id == api_id:
                return api
        return None

    def scan_api(self, api_id: str) -> Tuple[bool, Dict]:
        api = self.get_api(api_id)
        if api is None:
            return False, {"error": "api_not_found"}
        if api.hidden:
            return False, {"error": "api_not_visible"}

        # Inject noisy metadata: occasional stale/missing fields.
        noisy_version = api.version if self._rng.random() > 0.12 else self._rng.choice(VERSIONS)
        noisy_auth = api.auth_type if self._rng.random() > 0.10 else self._rng.choice(AUTH_TYPES)
        noisy_traffic = api.traffic_frequency
        if self._rng.random() < 0.18:
            noisy_traffic = round(max(0.01, noisy_traffic * self._rng.uniform(0.5, 1.6)), 2)

        result = {
            "api_id": api.api_id,
            "endpoint": api.endpoint,
            "version": noisy_version,
            "authentication_type": noisy_auth,
            "traffic_frequency": noisy_traffic,
            "last_updated": api.last_updated,
            "signal_strength": round(self._rng.uniform(0.55, 0.99), 2),
        }

        if self._rng.random() < 0.12:
            result["last_updated"] = "unknown"

        self._append_log(api.api_id, "scan", "success")
        discovered = self.maybe_discover_hidden_api()
        if discovered:
            result["new_discovery"] = discovered

        return True, result

    def run_security_test(self, api_id: str) -> Tuple[bool, Dict]:
        api = self.get_api(api_id)
        if api is None:
            return False, {"error": "api_not_found"}
        if api.hidden:
            return False, {"error": "api_not_visible"}

        flaky = self._rng.random() < 0.08
        if flaky:
            self._append_log(api.api_id, "security_test", "timeout")
            return False, {"error": "test_timeout"}

        detected: List[str] = []
        if api.vulnerabilities:
            for vuln in api.vulnerabilities:
                if self._rng.random() < 0.80:
                    detected.append(vuln)
            if not detected and self._rng.random() < 0.50:
                detected.append(self._rng.choice(api.vulnerabilities))
        else:
            if self._rng.random() < 0.12:
                detected.append("false_signal_low_confidence")

        confidence = round(self._rng.uniform(0.45, 0.98), 2)
        self._append_log(api.api_id, "security_test", "success")

        return True, {
            "api_id": api.api_id,
            "detected_vulnerabilities": detected,
            "confidence": confidence,
            "test_status": "complete",
        }

    def sample_logs(self, visible_api_ids: List[str], max_logs: int = 25) -> List[Dict]:
        scoped = [log for log in self.request_logs if log["api_id"] in visible_api_ids]
        return scoped[-max_logs:]

    def _append_log(self, api_id: str, event: str, status: str) -> None:
        self.request_logs.append(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "api_id": api_id,
                "event": event,
                "status": status,
            }
        )
