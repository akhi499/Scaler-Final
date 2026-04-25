"""GRPO/RLVR-style training entrypoint for ZombieShieldEnv.

This script uses TRL's GRPOTrainer with verifier-style rewards from the environment.
If GRPO is unavailable in the installed TRL version, it exits with a clear message.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import re
import sys
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.zombieshield_env import ZombieShieldEnv


def _build_prompt(obs: Dict) -> str:
    lines = [
        "You are an API security agent.",
        "Return exactly one action in one of these formats:",
        "scan_api(api_id)",
        "run_security_test(api_id)",
        "classify_api(api_id,ZOMBIE)",
        "classify_api(api_id,ACTIVE)",
        "block_api(api_id)",
        "ignore_api(api_id)",
        "escalate_api(api_id)",
        "",
        f"STEP={obs.get('step')} REMAINING={obs.get('steps_remaining')}",
        "VISIBLE_APIS:",
    ]
    for api in obs.get("apis", [])[:12]:
        lines.append(
            f"- {api['api_id']} observed={api.get('observed')} auth={api.get('authentication_type')} "
            f"ver={api.get('version')} traffic={api.get('traffic_frequency')} updated={api.get('last_updated')}"
        )
    lines.append("Action:")
    return "\n".join(lines)


def _extract_action(completion: str) -> str:
    completion = (completion or "").strip().splitlines()[0] if completion else ""
    match = re.search(r"(scan_api\([^\)]*\)|run_security_test\([^\)]*\)|classify_api\([^\)]*\)|block_api\([^\)]*\)|ignore_api\([^\)]*\)|escalate_api\([^\)]*\))", completion)
    if not match:
        return ""
    return match.group(1)


def _build_dataset(num_samples: int, min_apis: int, max_apis: int, seed: int):
    from datasets import Dataset

    rows: List[Dict] = []
    for i in range(num_samples):
        env_seed = seed + i
        env = ZombieShieldEnv(min_apis=min_apis, max_apis=max_apis, seed=env_seed)
        obs = env.reset()
        rows.append({"prompt": _build_prompt(obs), "env_seed": env_seed, "min_apis": min_apis, "max_apis": max_apis})

    return Dataset.from_list(rows)


def _format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        action = _extract_action(c)
        rewards.append(0.5 if action else -2.0)
    return rewards


def _env_reward(prompts, completions, env_seed, min_apis, max_apis, **kwargs):
    rewards = []
    for c, s, lo, hi in zip(completions, env_seed, min_apis, max_apis):
        action = _extract_action(c)
        if not action:
            rewards.append(-3.0)
            continue

        env = ZombieShieldEnv(min_apis=int(lo), max_apis=int(hi), seed=int(s))
        env.reset()
        _, reward, _, info = env.step(action)

        # Small extra bonus for producing valid-classification signals.
        shaped = float(reward)
        if info.get("status") == "ok":
            shaped += 0.5
        rewards.append(shaped)

    return rewards


def run(args: argparse.Namespace) -> None:
    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "GRPOTrainer is not available in the installed TRL build. "
            "Install a TRL version with GRPO support or use training/train_trl.py fallback path."
        ) from exc

    model = args.model_name
    tokenizer = None

    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model_obj, tokenizer = FastLanguageModel.from_pretrained(model_name=model, max_seq_length=1024)
            model = model_obj
        except Exception as exc:
            print(f"[warn] Unsloth unavailable; falling back to standard model loading: {exc}")

    dataset = _build_dataset(
        num_samples=args.dataset_samples,
        min_apis=args.min_apis,
        max_apis=args.max_apis,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg = GRPOConfig(
            output_dir=str(output_dir),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            logging_steps=args.logging_steps,
            max_completion_length=32,
        )
    except TypeError:
        cfg = GRPOConfig(output_dir=str(output_dir))

    trainer_kwargs = {
        "model": model,
        "reward_funcs": [_format_reward, _env_reward],
        "args": cfg,
        "train_dataset": dataset,
    }

    if tokenizer is not None:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()

    try:
        trainer.save_model(str(output_dir / "grpo_trained_model"))
    except Exception:
        pass

    meta = {
        "mode": "grpo_rlvr",
        "dataset_samples": args.dataset_samples,
        "epochs": args.epochs,
        "model_name": args.model_name,
    }
    (output_dir / "grpo_training_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRPO/RLVR policy for ZombieShieldEnv")
    parser.add_argument("--model-name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--min-apis", type=int, default=10)
    parser.add_argument("--max-apis", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-unsloth", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training_outputs_grpo")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
