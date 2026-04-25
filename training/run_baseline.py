"""Run baseline agents on ZombieShieldEnv using shared training utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.train_trl import RandomPolicy, HeuristicPolicy, evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--min-apis", type=int, default=10)
    parser.add_argument("--max-apis", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="training_outputs/baseline_results.json")
    args = parser.parse_args()

    env_kwargs = {"min_apis": args.min_apis, "max_apis": args.max_apis}
    payload = {
        "random": evaluate_policy(RandomPolicy(args.seed), args.episodes, env_kwargs, seed_base=args.seed + 100),
        "heuristic": evaluate_policy(HeuristicPolicy(args.seed + 7), args.episodes, env_kwargs, seed_base=args.seed + 200),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
