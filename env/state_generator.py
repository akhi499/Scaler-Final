"""State construction utilities for ZombieShieldEnv."""

from __future__ import annotations

from typing import Dict, List, Set

from simulator.api_simulator import APISimulator


class StateGenerator:
    def build(
        self,
        simulator: APISimulator,
        known_metadata: Dict[str, Dict],
        tested_results: Dict[str, Dict],
        visible_api_ids: Set[str],
        processed_api_ids: Set[str],
        step_count: int,
        max_steps: int,
    ) -> Dict:
        visible_sorted = sorted(visible_api_ids)
        apis: List[Dict] = []

        for api_id in visible_sorted:
            record = known_metadata.get(api_id, {"api_id": api_id, "observed": False})
            test_data = tested_results.get(api_id)

            entry = {
                "api_id": api_id,
                "observed": record.get("observed", False),
                "processed": api_id in processed_api_ids,
                "endpoint": record.get("endpoint", "unknown"),
                "version": record.get("version", "unknown"),
                "authentication_type": record.get("authentication_type", "unknown"),
                "traffic_frequency": record.get("traffic_frequency", "unknown"),
                "last_updated": record.get("last_updated", "unknown"),
                "last_signal_strength": record.get("signal_strength", "unknown"),
            }

            if test_data is not None:
                entry["security_test"] = {
                    "detected_vulnerabilities": test_data.get("detected_vulnerabilities", []),
                    "confidence": test_data.get("confidence", 0.0),
                }

            apis.append(entry)

        logs = simulator.sample_logs(visible_api_ids, max_logs=20)

        return {
            "step": step_count,
            "steps_remaining": max(0, max_steps - step_count),
            "visible_api_count": len(visible_api_ids),
            "apis": apis,
            "request_logs": logs,
            "available_actions": [
                "scan_api(api_id)",
                "classify_api(api_id, label)",
                "run_security_test(api_id)",
                "block_api(api_id)",
                "ignore_api(api_id)",
                "escalate_api(api_id)",
            ],
        }
