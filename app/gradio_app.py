"""Gradio demo for ZombieShieldEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import sys
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.zombieshield_env import ZombieShieldEnv


CUSTOM_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CUSTOM_ADAPTER_REPO = "akhi499/Zombie-API-Qwen-3B"
CUSTOM_ADAPTER_SUBFOLDER = "best_trl_model"
_CUSTOM_AGENT: Optional["CustomPPOAgent"] = None


@dataclass
class DemoState:
    env: ZombieShieldEnv
    observation: Dict
    done: bool
    total_reward: float
    decisions: List[Dict]
    reward_trace: List[float]


def candidate_actions(observation: Dict) -> List[str]:
    actions: List[str] = []
    apis = observation.get("apis", [])
    candidates = [api for api in apis if not api.get("processed", False)]
    if not candidates:
        candidates = apis

    for api in candidates:
        api_id = api["api_id"]
        actions.extend(
            [
                f"scan_api({api_id})",
                f"run_security_test({api_id})",
                f"classify_api({api_id},ZOMBIE)",
                f"classify_api({api_id},ACTIVE)",
                f"block_api({api_id})",
                f"ignore_api({api_id})",
                f"escalate_api({api_id})",
            ]
        )
    return actions[:120]


class EpsilonGreedyCheckpointPolicy:
    """Inference-only policy loader for fallback checkpoint JSON."""

    def __init__(self, q_table: Dict[Tuple[str, str, int], float], seed: int = 42):
        self.q = q_table
        self.rng = random.Random(seed)

    @staticmethod
    def _parse_action(action: str) -> Tuple[str, str]:
        action_type = action.split("(", 1)[0]
        api_id = action[action.find("(") + 1 : action.rfind(")")].split(",", 1)[0]
        return action_type, api_id

    @staticmethod
    def _state_bucket(obs: Dict) -> str:
        remain = int(obs.get("steps_remaining", 0))
        if remain > 20:
            return "early"
        if remain > 8:
            return "mid"
        return "late"

    @staticmethod
    def _risk_bucket(api: Dict) -> int:
        score = 0
        if api.get("authentication_type") == "none":
            score += 2
        if str(api.get("version", "")).lower() in {"legacy", "v1"}:
            score += 1
        if str(api.get("last_updated", "unknown")) != "unknown" and str(api.get("last_updated")) < "2024-01-01":
            score += 1
        if api.get("security_test", {}).get("detected_vulnerabilities"):
            score += 2
        if not api.get("observed", False):
            score -= 1
        if api.get("processed", False):
            score -= 2

        if score <= 0:
            return 0
        if score <= 2:
            return 1
        return 2

    def _action_key(self, obs: Dict, action: str) -> Tuple[str, str, int]:
        state_bucket = self._state_bucket(obs)
        action_type, api_id = self._parse_action(action)
        api_lookup = {api["api_id"]: api for api in obs.get("apis", [])}
        risk_bucket = self._risk_bucket(api_lookup.get(api_id, {}))
        return (state_bucket, action_type, risk_bucket)

    def act(self, obs: Dict, actions: List[str]) -> str:
        if not actions:
            return "scan_api(api_0)"

        # Keep exploration behavior from training script: scan unknowns first.
        for api in obs.get("apis", []):
            if not api.get("observed", False) and not api.get("processed", False):
                candidate = f"scan_api({api['api_id']})"
                if candidate in actions:
                    return candidate

        best_action = None
        best_q = float("-inf")
        for action in actions:
            key = self._action_key(obs, action)
            q = self.q.get(key, 0.0)
            if q > best_q:
                best_q = q
                best_action = action
        return best_action or self.rng.choice(actions)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, seed: int = 42) -> "EpsilonGreedyCheckpointPolicy":
        payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        q_payload = payload.get("q", {})
        restored_q: Dict[Tuple[str, str, int], float] = {}
        for raw_key, value in q_payload.items():
            parts = str(raw_key).split("|")
            if len(parts) != 3:
                continue
            state_bucket, action_type, risk_bucket = parts[0], parts[1], parts[2]
            try:
                restored_q[(state_bucket, action_type, int(risk_bucket))] = float(value)
            except Exception:
                continue
        return cls(q_table=restored_q, seed=seed)


class HeuristicAgent:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def choose_action(self, obs: Dict) -> str:
        apis = obs.get("apis", [])
        if not apis:
            return "scan_api(api_0)"

        active_apis = [a for a in apis if not a.get("processed", False)]
        if not active_apis:
            return f"ignore_api({apis[0]['api_id']})"

        api = self.rng.choice(active_apis)
        api_id = api["api_id"]

        if not api.get("observed", False):
            return f"scan_api({api_id})"

        sec = api.get("security_test")
        if sec is None and self.rng.random() < 0.55:
            return f"run_security_test({api_id})"

        hints = 0
        if api.get("authentication_type") == "none":
            hints += 1
        if str(api.get("version", "")).lower() in {"legacy", "v1"}:
            hints += 1
        if str(api.get("last_updated", "unknown")) != "unknown" and api.get("last_updated", "") < "2024-01-01":
            hints += 1
        if sec and sec.get("detected_vulnerabilities"):
            hints += 2

        if hints >= 2:
            if self.rng.random() < 0.7:
                return f"classify_api({api_id},ZOMBIE)"
            return f"block_api({api_id})"

        if self.rng.random() < 0.45:
            return f"classify_api({api_id},ACTIVE)"
        return f"ignore_api({api_id})"


class CustomPPOAgent:
    """Inference agent that loads Qwen base model + LoRA adapter from Hugging Face."""

    def __init__(
        self,
        base_model: str = CUSTOM_BASE_MODEL,
        adapter_repo: str = CUSTOM_ADAPTER_REPO,
        adapter_subfolder: str = CUSTOM_ADAPTER_SUBFOLDER,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available():
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base, adapter_repo, subfolder=adapter_subfolder)
            self.device = "cuda"
        else:
            base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
            model = PeftModel.from_pretrained(base, adapter_repo, subfolder=adapter_subfolder)
            self.device = "cpu"
            model.to(self.device)

        self.model = model.eval()

    @staticmethod
    def _pick_valid_action(raw: str, actions: List[str]) -> str:
        cleaned = raw.strip()
        if cleaned in actions:
            return cleaned
        for act in actions:
            if cleaned.startswith(act.split("(", 1)[0]):
                return act
        return random.choice(actions)

    def choose_action(self, obs: Dict) -> str:
        actions = candidate_actions(obs)
        if not actions:
            return "scan_api(api_0)"

        api_lines = []
        for api in obs.get("apis", [])[:12]:
            api_lines.append(
                f"- {api['api_id']} endpoint={api.get('endpoint')} version={api.get('version')} "
                f"auth={api.get('authentication_type')} traffic={api.get('traffic_frequency')} "
                f"updated={api.get('last_updated')}"
            )

        action_list = "\n".join([f"* {a}" for a in actions[:30]])
        prompt = (
            "You are a zombie API defense agent. Pick exactly one action string from ACTIONS.\n"
            "Goal: detect zombie APIs, avoid false positives, and mitigate correctly.\n"
            f"STEP={obs.get('step')} REMAINING={obs.get('steps_remaining')}\n"
            "VISIBLE_APIS:\n" + "\n".join(api_lines) + "\n"
            f"ACTIONS:\n{action_list}\n"
            "Answer with one action only:"
        )

        toks = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            return_attention_mask=True,
        )
        input_ids = toks.input_ids.to(self.device)
        attention_mask = toks.attention_mask.to(self.device)

        with self.torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = generated[:, input_ids.shape[1]:]
        decoded = self.tokenizer.decode(response_ids[0], skip_special_tokens=True).strip().splitlines()
        raw_action = decoded[0] if decoded else ""
        return self._pick_valid_action(raw_action, actions)


def _get_custom_agent() -> CustomPPOAgent:
    global _CUSTOM_AGENT
    if _CUSTOM_AGENT is None:
        _CUSTOM_AGENT = CustomPPOAgent()
    return _CUSTOM_AGENT


def init_demo(seed: int = 42) -> DemoState:
    env = ZombieShieldEnv(min_apis=10, max_apis=20, seed=seed)
    obs = env.reset()
    return DemoState(env=env, observation=obs, done=False, total_reward=0.0, decisions=[], reward_trace=[])


def _choose_action(obs: Dict, agent_mode: str, checkpoint_path: str, step_idx: int) -> Tuple[str, str]:
    if agent_mode == "Custom PPO Trained Agent":
        try:
            agent = _get_custom_agent()
            return agent.choose_action(obs), "Custom PPO Trained Agent"
        except Exception as exc:
            heuristic = HeuristicAgent(seed=42 + step_idx)
            return heuristic.choose_action(obs), f"heuristic_fallback_custom_load_error:{type(exc).__name__}"

    if agent_mode == "trained_checkpoint":
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            policy = EpsilonGreedyCheckpointPolicy.from_checkpoint(str(ckpt), seed=42)
            actions = candidate_actions(obs)
            return policy.act(obs, actions), "trained_checkpoint"
        # Fallback gracefully if checkpoint is missing.
        heuristic = HeuristicAgent(seed=42 + step_idx)
        return heuristic.choose_action(obs), "heuristic_fallback_no_checkpoint"

    heuristic = HeuristicAgent(seed=42 + step_idx)
    return heuristic.choose_action(obs), "heuristic"


def _table_from_observation(obs: Dict) -> List[List]:
    rows: List[List] = []
    for api in obs.get("apis", []):
        rows.append(
            [
                api["api_id"],
                api.get("endpoint"),
                api.get("version"),
                api.get("authentication_type"),
                api.get("traffic_frequency"),
                api.get("last_updated"),
                ",".join(api.get("security_test", {}).get("detected_vulnerabilities", [])),
            ]
        )
    return rows


def _reward_plot(reward_trace: List[float]):
    fig, ax = plt.subplots(figsize=(7, 3))
    if reward_trace:
        ax.plot(range(1, len(reward_trace) + 1), reward_trace, color="#0b7285", marker="o", markersize=3)
    ax.set_title("Reward Progression")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def reset_env(seed: int, agent_mode: str, checkpoint_path: str) -> Tuple[DemoState, List[List], str, str, object]:
    state = init_demo(seed=seed)
    table = _table_from_observation(state.observation)
    checkpoint_note = ""
    if agent_mode == "trained_checkpoint" and not Path(checkpoint_path).exists():
        checkpoint_note = " (checkpoint missing, will fallback to heuristic)"
    status = (
        f"Episode started. visible_apis={state.observation['visible_api_count']} "
        f"agent_mode={agent_mode}{checkpoint_note}"
    )
    decisions = "No decisions yet."
    return state, table, status, decisions, _reward_plot(state.reward_trace)


def step_once(state: DemoState, agent_mode: str, checkpoint_path: str):
    if state.done:
        table = _table_from_observation(state.observation)
        return state, table, "Episode already finished. Click Reset.", _decisions_text(state.decisions), _reward_plot(state.reward_trace)

    action, resolved_mode = _choose_action(state.observation, agent_mode, checkpoint_path, len(state.decisions))
    next_obs, reward, done, info = state.env.step(action)

    state.observation = next_obs
    state.done = done
    state.total_reward += reward
    state.reward_trace.append(reward)
    state.decisions.append({"action": action, "reward": round(reward, 2), "info": info})

    table = _table_from_observation(state.observation)
    status = (
        f"step={info.get('step')} reward={reward:.2f} total_reward={state.total_reward:.2f} "
        f"done={done} accuracy={info.get('terminal_accuracy', 'n/a')} agent_mode={resolved_mode}"
    )
    decisions = _decisions_text(state.decisions)

    return state, table, status, decisions, _reward_plot(state.reward_trace)


def autoplay(state: DemoState, max_steps: int, agent_mode: str, checkpoint_path: str):
    for _ in range(max_steps):
        if state.done:
            break
        state, _, _, _, _ = step_once(state, agent_mode, checkpoint_path)

    table = _table_from_observation(state.observation)
    final_info = state.decisions[-1]["info"] if state.decisions else {}
    status = (
        f"Autoplay done. steps={len(state.decisions)} total_reward={state.total_reward:.2f} "
        f"terminal_accuracy={final_info.get('terminal_accuracy', 'n/a')} "
        f"vulnerabilities={final_info.get('terminal_vulnerabilities_detected', 'n/a')} "
        f"agent_mode={agent_mode}"
    )
    return state, table, status, _decisions_text(state.decisions), _reward_plot(state.reward_trace)


def _decisions_text(decisions: List[Dict]) -> str:
    if not decisions:
        return "No decisions yet."

    lines: List[str] = []
    for idx, d in enumerate(decisions[-20:], start=max(1, len(decisions) - 19)):
        lines.append(f"{idx}. {d['action']} -> reward={d['reward']}")
    return "\n".join(lines)


def training_summary_text() -> str:
    candidates = []
    preferred = Path("training_outputs/training_summary.json")
    if preferred.exists():
        candidates.append(preferred)

    candidates.extend(sorted(Path(".").glob("training_outputs*/training_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True))
    if not candidates:
        return "No training summary found yet. Run training/train_trl.py first."

    summary_path = candidates[0]
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rendered = json.dumps(payload, indent=2)
    if summary_path != preferred:
        rendered = f"// Loaded from: {summary_path.as_posix()}\n{rendered}"
    return rendered


def comparison_plot_path() -> str | None:
    preferred = Path("training_outputs/baseline_comparison.png")
    if preferred.exists():
        return str(preferred)

    candidates = sorted(Path(".").glob("training_outputs*/baseline_comparison.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def reload_artifacts() -> Tuple[str, str | None]:
    return training_summary_text(), comparison_plot_path()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ZombieShieldEnv Demo") as demo:
        gr.Markdown(
            "# ZombieShieldEnv\n"
            "Interactive demo for zombie API discovery, classification, and mitigation.\n\n"
            "Use this UI to show what the agent observes, what action it takes each step, and how reward evolves.\n\n"
            "**Tip:** choose `trained_checkpoint` and point to `training_outputs/policy_checkpoint.json` "
            "after training to demo learned behavior in Space."
        )

        state = gr.State(init_demo())

        with gr.Row():
            seed = gr.Number(value=42, label="Seed", precision=0)
            agent_mode = gr.Dropdown(
                choices=["heuristic", "trained_checkpoint", "Custom PPO Trained Agent"],
                value="heuristic",
                label="Agent Mode",
            )
            checkpoint_path = gr.Textbox(
                value="training_outputs/policy_checkpoint.json",
                label="Checkpoint path",
            )
            refresh_btn = gr.Button("Reload Artifacts")
            reset_btn = gr.Button("Reset Episode")
            step_btn = gr.Button("Step Once")
            auto_btn = gr.Button("Auto Play")
            auto_steps = gr.Slider(minimum=5, maximum=120, value=40, step=1, label="Auto steps")

        api_table = gr.Dataframe(
            headers=["api_id", "endpoint", "version", "auth", "traffic", "last_updated", "detected_vulns"],
            datatype=["str", "str", "str", "str", "str", "str", "str"],
            row_count=(10, "dynamic"),
            label="Visible APIs",
        )
        status = gr.Textbox(label="Status")
        decisions = gr.Textbox(label="Recent Decisions", lines=12)
        reward_plot = gr.Plot(label="Reward Progression")
        summary = gr.Code(label="Latest Training Summary", language="json", value=training_summary_text())
        comparison_img = gr.Image(label="Baseline vs Trained Comparison", value=comparison_plot_path())

        reset_btn.click(
            reset_env,
            inputs=[seed, agent_mode, checkpoint_path],
            outputs=[state, api_table, status, decisions, reward_plot],
        )
        step_btn.click(
            step_once,
            inputs=[state, agent_mode, checkpoint_path],
            outputs=[state, api_table, status, decisions, reward_plot],
        )
        auto_btn.click(
            autoplay,
            inputs=[state, auto_steps, agent_mode, checkpoint_path],
            outputs=[state, api_table, status, decisions, reward_plot],
        )
        refresh_btn.click(
            reload_artifacts,
            inputs=[],
            outputs=[summary, comparison_img],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
