"""Evaluate random/heuristic/trained policies across easy/medium/hard tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.train_trl import (
    HeuristicPolicy,
    RandomPolicy,
    TRLPPOPolicy,
    evaluate_across_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Difficulty benchmark for ZombieShieldEnv")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--trained-model-dir", type=str, default="")
    parser.add_argument("--output", type=str, default="training_outputs/task_benchmark_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random_policy = RandomPolicy(seed=args.seed + 11)
    heuristic_policy = HeuristicPolicy(seed=args.seed + 17)

    trained_dir = args.trained_model_dir.strip()
    if not trained_dir:
        trained_dir = "training_outputs/trained_model"

    trained_policy = TRLPPOPolicy(
        model_name=args.model_name,
        seed=args.seed,
        load_from=trained_dir,
    )

    payload = {
        "episodes": args.episodes,
        "random": evaluate_across_tasks(
            random_policy, episodes=args.episodes, seed_base=args.seed + 1000, deterministic=True
        ),
        "heuristic": evaluate_across_tasks(
            heuristic_policy, episodes=args.episodes, seed_base=args.seed + 2000, deterministic=True
        ),
        "trained": evaluate_across_tasks(
            trained_policy, episodes=args.episodes, seed_base=args.seed + 3000, deterministic=True
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

