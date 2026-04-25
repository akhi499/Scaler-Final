---
<<<<<<< HEAD
title: Zombie API
emoji: 🦀
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.13.0
=======
title: ZombieShieldEnv
emoji: "🧟"
colorFrom: blue
colorTo: green
sdk: gradio
>>>>>>> e7f48e1 (Deploy ZombieShieldEnv to HF Space)
app_file: app.py
pinned: false
---

<<<<<<< HEAD
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
=======
# ZombieShieldEnv

ZombieShieldEnv is an OpenEnv-compatible environment for training LLM agents to discover, classify, test, and mitigate zombie APIs (undocumented/stale/risky endpoints).

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
- Fallback: epsilon-greedy Q-learning (keeps prototype runnable when TRL is unavailable)
- Training loop interacts with environment directly (not static data)

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
For CPU-only fallback run:
```bash
python training/train_trl.py --episodes 80 --eval-episodes 12 --no-prefer-trl --output-dir training_outputs
```
To resume a saved fallback policy checkpoint:
```bash
python training/train_trl.py --episodes 40 --no-prefer-trl --load-checkpoint training_outputs/policy_checkpoint.json --output-dir training_outputs_resume
```
To run GRPO/RLVR training:
```bash
python training/train_grpo_rlvr.py --dataset-samples 256 --epochs 1 --output-dir training_outputs_grpo
```

Expected logs every few episodes:
- `episode=... reward=... accuracy=...`
- final JSON summary including `agent_pretrain`, `agent_posttrain`, and `improvement`.

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
huggingface-cli login
huggingface-cli repo create <your-space-name> --type space --space_sdk gradio
git remote add space https://huggingface.co/spaces/<hf-username>/<your-space-name>
git push space HEAD:main
```

After deploy, verify:
- Space loads without dependency errors
- Demo UI shows `Visible APIs`, `Recent Decisions`, and reward progression
- `training_outputs/training_summary.json` is visible in the demo section

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
  - Hosted on Hugging Face Spaces: **pending until you push Space**
  - Mini-blog / <2 min video / slides linked in README: **pending links**
- Evidence artifacts present:
  - `training_outputs/reward_curve.png`
  - `training_outputs/accuracy_curve.png`
  - `training_outputs/vulnerabilities_curve.png`
  - `training_outputs/baseline_comparison.png`
  - `training_outputs/training_summary.json`

## Debugging
- `ModuleNotFoundError: trl`: install dependencies or use `--no-prefer-trl`.
- `ModuleNotFoundError: gradio`: run `pip install -r requirements.txt` (or `requirements-colab.txt` in Colab).
- Flat/noisy curves: increase episodes (`120+`) and keep seed fixed.
- Weak improvement vs baseline: increase eval episodes and check reward penalties for over-blocking.
- `GRPOTrainer unavailable`: use `requirements-colab.txt` or fallback training via `train_trl.py`.

## Why It Matters
This environment simulates realistic API security operations under uncertainty and creates measurable RL feedback for LLM behavior improvement.

It is useful for training agents that can perform dependable API governance work in enterprise settings.

>>>>>>> e7f48e1 (Deploy ZombieShieldEnv to HF Space)
