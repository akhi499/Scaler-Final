---
title: ZombieShieldEnv
emoji: "🧟"
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# ZombieShieldEnv

ZombieShieldEnv is an OpenEnv-compatible environment for training LLM agents to discover, classify, test, and mitigate zombie APIs (undocumented/stale/risky endpoints).

## Submission Links (Judges)
- Hugging Face Space: https://huggingface.co/spaces/akhi499/Zombie-API
- Mini-blog (draft markdown in repo): `BLOG_SUBMISSION.md`

## Direct Evidence Links (UI-Independent)
If the Space UI cache is stale, use these direct artifact links in the Files tab:
- Training summary JSON: https://huggingface.co/spaces/akhi499/Zombie-API/blob/main/training_outputs/training_summary.json
- Baseline results JSON: https://huggingface.co/spaces/akhi499/Zombie-API/blob/main/training_outputs/baseline_results.json
- Policy checkpoint JSON: https://huggingface.co/spaces/akhi499/Zombie-API/blob/main/training_outputs/policy_checkpoint.json

Judge fallback note:
- If the `Latest Training Summary` widget says no file found, the above links are the source of truth for evaluation.

## 2-Minute Judge Story Flow
1. Problem: zombie APIs remain reachable with stale auth/versions and hidden vulnerabilities.
2. Environment: partially observable API ecosystem with step-wise actions (`scan`, `classify`, `test`, `block`, `ignore`, `escalate`).
3. Reward design: multi-signal shaping with anti-gaming constraints.
4. Training: TRL PPO (`training/train_trl.py`); older Q-table fallback lives on branch `backup/pre-trl-fix`.
5. Evidence: pre-train vs post-train metrics and baseline comparisons.
6. Why this matters: reproducible RL environment for enterprise API governance behavior.

## Hackathon Theme Fit
Primary fit: **Theme #3 - World Modeling / Professional Tasks**.

Why this fits:
- The agent interacts with a dynamic API ecosystem (tools/APIs with noisy signals).
- The world is partially observable and requires persistent state updates.
- The task is real professional security work: discovery, risk triage, and mitigation.

## Problem
Zombie APIs are hard to detect because telemetry is incomplete and noisy. Many stay reachable with weak auth, stale versions, and hidden vulnerabilities.

This environment trains an LLM to make sequential operational decisions under uncertainty:
- discover hidden APIs
- classify ACTIVE vs ZOMBIE
- run security tests
- choose mitigation actions (block/ignore/escalate)

## Environment
Class: `ZombieShieldEnv(Environment)` in `env/zombieshield_env.py`.

Implemented API:
- `reset()`
- `step(action)`
- `state()`

### Observation (partial observability)
`state()` returns JSON-like data:
- visible APIs only (hidden APIs may appear after scans)
- per-API metadata: endpoint, version, auth type, traffic, last updated
- optional security test outputs
- request logs

### Actions
- `scan_api(api_id)`
- `classify_api(api_id,label)` where label is `ACTIVE` or `ZOMBIE`
- `run_security_test(api_id)`
- `block_api(api_id)`
- `ignore_api(api_id)`
- `escalate_api(api_id)`

### Reward design
Dense reward shaping in `env/reward_engine.py`:
- classify zombie correctly: `+10`
- miss zombie: `-15`
- false positive: `-5`
- detect real vuln via test: `+8`
- block confirmed zombie: `+15`
- block valid API: `-10`
- efficiency bonus at episode end
- repeated/invalid/random actions penalized

Anti-gaming controls:
- one-time reward credit per API/event
- repeated actions incur penalties
- end-of-episode penalties for unhandled zombie APIs
- per-API action budget guardrails to prevent infinite-loop exploitation
- simulated test timeouts and invalid-action penalties

## Training Pipeline
Script: `training/train_trl.py`

Note: latest `trl` versions changed APIs. For hackathon Colab reruns, use `requirements-colab.txt` (pins `trl==0.8.6` for PPOTrainer compatibility).

- Preferred: Hugging Face TRL PPO with TinyLlama
- Training loop interacts with environment directly (not static data)
- Difficulty benchmark includes deterministic `easy`, `medium`, `hard` task grading (0.0-1.0)

The script outputs:
- reward/accuracy/vulnerability curves
- precision/recall/F1/labeled-fraction curve
- baseline vs pretrain vs posttrain comparison
- JSON summary with improvement deltas

### GRPO/RLVR path
Script: `training/train_grpo_rlvr.py`

- Uses TRL `GRPOTrainer` when available
- Uses verifier-style reward functions that call the environment directly
- Optional `--use-unsloth` flag for efficiency if Unsloth is installed

## Baselines
Script: `training/run_baseline.py`

Includes:
- random baseline agent
- rule-based heuristic baseline

Used to satisfy "trained vs random/untrained baseline" judging requirement.

## Project Structure
```text
zombieshield_env/
├── env/
│   ├── zombieshield_env.py
│   ├── state_generator.py
│   └── reward_engine.py
├── simulator/
│   └── api_simulator.py
├── training/
│   ├── train_trl.py
│   ├── train_grpo_rlvr.py
│   ├── run_baseline.py
│   └── test_env.py
├── app/
│   ├── gradio_app.py
│   └── server.py
├── openenv.yaml
├── requirements.txt
├── requirements-colab.txt
├── requirements-server.txt
├── Dockerfile
└── README.md
```

## Setup (Local / Colab)
Recommended Python: `3.10-3.12`.

```bash
pip install -U pip
pip install -r requirements.txt
# For Colab TRL PPO compatibility:
pip install -r requirements-colab.txt
```

Colab notebook:
- `notebooks/colab_train.ipynb`

`requirements.txt` pins OpenEnv latest used here:
- `openenv-core==0.2.3`

## Local Testing Guide
### 1) Environment smoke test
```bash
python training/test_env.py
```
Expected:
- `reset_ok ...`
- several step logs
- `multi_episode_ok True`

### 2) Baseline test
```bash
python training/run_baseline.py --episodes 20 --output training_outputs/baseline_results.json
```
Expected:
- JSON with `random` and `heuristic` metrics.

### 3) Training run
```bash
python training/train_trl.py --episodes 80 --eval-episodes 12 --output-dir training_outputs
```
To run GRPO/RLVR training:
```bash
python training/train_grpo_rlvr.py --dataset-samples 256 --epochs 1 --output-dir training_outputs_grpo
```
To run explicit easy/medium/hard comparison after training:
```bash
python training/evaluate_tasks.py --model-name Qwen/Qwen2.5-3B-Instruct --trained-model-dir training_outputs/trained_model --episodes 12 --output training_outputs/task_benchmark_summary.json
```

Expected logs every few episodes:
- `episode=... reward=... accuracy=...`
- final JSON summary including `agent_pretrain`, `agent_posttrain`, and `improvement`.
- per-difficulty metrics in `difficulty_eval` with deterministic task scores.

### 4) Verify learning
Look in `training_outputs/training_summary.json`:
- `improvement.reward_delta > 0` is ideal
- `improvement.accuracy_delta > 0` is ideal
- trained agent should beat random baseline on mean reward

### 5) Plot artifacts (for judging)
Generated in `training_outputs/`:
- `reward_curve.png`
- `accuracy_curve.png`
- `vulnerabilities_curve.png`
- `classification_metrics_curve.png`
- `baseline_comparison.png`

## Gradio Demo
```bash
python app/gradio_app.py
```
What to show in demo:
- table of visible APIs (what agent sees)
- decision log (what agent does)
- reward progression (how behavior evolves)
- baseline comparison image + training summary

## FastAPI + Docker Deployment
Run local API server:
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```
API endpoints:
- `GET /health`
- `POST /reset`
- `GET /state`
- `POST /step`

Build and run Docker container:
```bash
docker build -t zombieshield-env .
docker run --rm -p 8000:8000 zombieshield-env
```

## Hugging Face Spaces Deployment
1. Create a Gradio Space.
2. Push this repository to that Space (or upload all files).
3. Keep app file as `app.py` (root entrypoint for Spaces).
4. Ensure `requirements.txt` is present.
5. Include links in README to:
   - Space URL
   - mini-blog or <2 min video
   - any slides/writeup

CLI deployment (recommended):
```bash
pip install -U huggingface_hub
hf auth login
hf repo create <your-space-name> --repo-type space --space_sdk gradio
# Avoid git binary restrictions by using API upload with excludes:
hf upload <hf-username>/<your-space-name> . . --repo-type space \
  --exclude "*.pdf" "*.png" "*.jpg" "*.jpeg" "*.gif" "**/__pycache__/**" "training_outputs*/**"
```

After deploy, verify:
- Space loads without dependency errors
- Demo UI shows `Visible APIs`, `Recent Decisions`, and reward progression
- `training_outputs/training_summary.json` is visible in the demo section

### Train in Colab, then use trained policy in Space
1. In Colab:
```bash
pip install -U pip
pip install -r requirements-colab.txt
python training/train_trl.py --episodes 120 --eval-episodes 20 --no-prefer-trl --output-dir training_outputs
```
2. Upload trained artifacts to your Space:
```bash
hf upload akhi499/Zombie-API training_outputs/policy_checkpoint.json training_outputs/policy_checkpoint.json --repo-type space
hf upload akhi499/Zombie-API training_outputs/training_summary.json training_outputs/training_summary.json --repo-type space
```
3. In Space UI:
   - Set `Agent Mode` to `trained_checkpoint`
   - Keep checkpoint path as `training_outputs/policy_checkpoint.json`
   - Reset + run episode to compare behavior vs `heuristic`

## Hackathon Submission Checklist
- OpenEnv latest release used: `openenv-core==0.2.3`
- OpenEnv/Gym-style methods: `reset`, `step`, `state`
- Working training script using TRL (or fallback for reproducibility)
- Real training evidence: plots + summary JSON
- Baseline comparison included
- README explains Problem, Environment, Results, Why it matters
- Space-hosted runnable demo + linked external materials

## Judge Round Readiness (Current Status)
- Theme fit: **Theme #3.1 Professional Tasks (World Modeling)**
- Minimum requirements:
  - OpenEnv latest: **met** (`openenv-core==0.2.3`)
  - Training script with TRL/Unsloth path: **met** (`training/train_trl.py`, `training/train_grpo_rlvr.py`)
  - Hosted on Hugging Face Spaces: **met** (https://huggingface.co/spaces/akhi499/Zombie-API)
  - Mini-blog / <2 min video / slides linked in README: **pending links**
- Evidence artifacts present:
  - `training_outputs/training_summary.json`
  - `training_outputs/baseline_results.json`

## Measured Improvement (Latest Run)
From `training_outputs/training_summary.json` (last uploaded Space run; re-run training after PPO-only script push to refresh):

| Metric | Pretrain | Posttrain | Delta |
|---|---:|---:|---:|
| Mean reward | -212.05 | -186.00 | +26.05 |
| Accuracy | 0.4058 | 0.4655 | +0.0596 |
| Precision | 0.3342 | 0.4655 | +0.1313 |
| Recall | 0.1959 | 0.2699 | +0.0740 |
| F1 | 0.2430 | 0.3299 | +0.0869 |
| Vulnerabilities (detected count) | 5.3 | 0.8 | -4.5 |

Baseline reference from `training_outputs/baseline_results.json`:
- Random baseline mean reward: `-218.20`
- Heuristic baseline mean reward: `81.44`
- Random baseline mean F1: `0.2992`
- Heuristic baseline mean F1: `0.6521`

## Reinforcement Learning & Evaluation
We trained a custom PPO policy on top of `Qwen/Qwen2.5-3B-Instruct` to improve step-wise zombie API detection, mitigation quality, and false-positive control in a partially observable security environment. The training loop optimizes actions such as scanning, classification, security testing, and blocking with dense reward signals tied to real operational outcomes. To make this practical on constrained hardware, we used PEFT/LoRA and trained adapter weights on a single L4 GPU, then exported the best adapter checkpoint for efficient deployment and inference in Hugging Face Spaces.

![Reward Curve](https://huggingface.co/akhi499/Zombie-API-Qwen-3B/resolve/main/reward_curve.png)
![Baseline Comparison](https://huggingface.co/akhi499/Zombie-API-Qwen-3B/resolve/main/baseline_comparison.png)
![Difficulty Task Score Comparison](https://huggingface.co/akhi499/Zombie-API-Qwen-3B/resolve/main/difficulty_task_score_comparison.png)

## Submission Status
Current recommendation: **Ready to submit** after you re-run training on the PPO-only script and refresh `training_outputs/*` if needed.

Reasons:
- Core minimum criteria are met (OpenEnv usage, training pipeline, Space-hosted runnable demo).
- Direct artifact links are available in README if the UI widget cache is stale.

Final pre-submit checks:
1. Replace all `TODO_ADD_LINK` placeholders with your final public links (video/blog/slides).
2. Verify once in Space: `Agent Mode = trained_checkpoint`, run one full episode.
3. Confirm `training_outputs/training_summary.json` opens from the direct link.

## Debugging
- `ModuleNotFoundError: trl` / import errors: install `pip install "trl>=0.7.10,<0.9.0"` (see `requirements.txt`). Training no longer supports `--no-prefer-trl`.
- `ModuleNotFoundError: gradio`: run `pip install -r requirements.txt` (or `requirements-colab.txt` in Colab).
- Flat/noisy curves: increase episodes (`120+`) and keep seed fixed.
- Weak improvement vs baseline: increase eval episodes and check reward penalties for over-blocking.
- `GRPOTrainer unavailable`: use `requirements-colab.txt` or `training/train_trl.py` (TRL PPO). For historical Q-table training use branch `backup/pre-trl-fix`.

## Why It Matters
This environment simulates realistic API security operations under uncertainty and creates measurable RL feedback for LLM behavior improvement.

It is useful for training agents that can perform dependable API governance work in enterprise settings.
