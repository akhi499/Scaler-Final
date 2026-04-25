# ZombieShieldEnv: Training LLM Agents for Zombie API Defense

## Problem
Zombie APIs (undocumented, stale, or weakly protected endpoints) are common in real systems and difficult to manage with static rules. They create security and governance risk because telemetry is noisy, ownership is unclear, and risk signals are partial.

We built `ZombieShieldEnv`, an OpenEnv-compatible RL environment where an agent must discover APIs, classify risk, run tests, and choose mitigation actions under uncertainty.

## Why this environment is interesting
- It models a realistic professional workflow instead of a toy puzzle.
- The world is partially observable, so the agent must gather information before acting.
- The task requires sequential decisions with consequences (scan -> classify -> test -> mitigate).

## Environment Design
The environment implements standard OpenEnv/Gym-style methods:
- `reset()`
- `step(action)`
- `state()`

### Action space
- `scan_api(api_id)`
- `classify_api(api_id,ACTIVE|ZOMBIE)`
- `run_security_test(api_id)`
- `block_api(api_id)`
- `ignore_api(api_id)`
- `escalate_api(api_id)`

### Reward design
Reward is multi-signal, not monolithic:
- Classification quality (true positives vs false positives/misses)
- Vulnerability detection signal
- Mitigation quality (blocking confirmed zombies vs blocking healthy APIs)
- Action validity/repetition penalties
- Terminal penalties for missed zombies + efficiency bonus

We also added anti-gaming logic to reduce reward hacking:
- one-time reward credit per API/event
- penalties for repeated/invalid action patterns
- per-API action budget guardrails
- terminal shaping tied to recall/F1/labeled-fraction to discourage low-coverage conservative collapse

## Training setup
We support two paths:
1. TRL/GRPO path (`training/train_grpo_rlvr.py`) for verifier-style RL experiments
2. Reliable fallback path (`training/train_trl.py` with `epsilon_q_fallback`) for deterministic hackathon reproducibility

For this submission, we used the fallback training path and evaluated:
- random baseline
- heuristic baseline
- pretrain policy
- posttrain policy

## Key Results
From the latest tuned run (`training_outputs/training_summary.json`):
- `reward_delta`: **+26.34**
- `accuracy_delta`: **+0.109**
- `precision_delta`: **+0.181**
- `recall_delta`: **+0.101**
- `f1_delta`: **+0.131**

This shows measurable learning improvement across reward and core classification metrics.

## Deployment
The environment and demo are deployed on Hugging Face Spaces:
- https://huggingface.co/spaces/akhi499/Zombie-API

Judges can inspect artifact files directly:
- `training_outputs/training_summary.json`
- `training_outputs/baseline_results.json`
- `training_outputs/policy_checkpoint.json`

## What we learned
A major practical lesson was that reward-only optimization can hide failure modes (for example, conservative low-coverage behavior). Multi-metric monitoring and anti-gaming reward shaping were essential to align optimization with real task quality.

## Next steps
- Run larger TRL/GRPO experiments with more compute credits
- Compare multiple seeds and report confidence intervals
- Add side-by-side policy comparison view in Space for clearer live demos
