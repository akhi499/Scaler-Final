"""Gradio demo for ZombieShieldEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import sys
from typing import Dict, List, Tuple

import gradio as gr
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.zombieshield_env import ZombieShieldEnv


@dataclass
class DemoState:
    env: ZombieShieldEnv
    observation: Dict
    done: bool
    total_reward: float
    decisions: List[Dict]
    reward_trace: List[float]


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


def init_demo(seed: int = 42) -> DemoState:
    env = ZombieShieldEnv(min_apis=10, max_apis=20, seed=seed)
    obs = env.reset()
    return DemoState(env=env, observation=obs, done=False, total_reward=0.0, decisions=[], reward_trace=[])


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


def reset_env(seed: int) -> Tuple[DemoState, List[List], str, str, object]:
    state = init_demo(seed=seed)
    table = _table_from_observation(state.observation)
    status = f"Episode started. visible_apis={state.observation['visible_api_count']}"
    decisions = "No decisions yet."
    return state, table, status, decisions, _reward_plot(state.reward_trace)


def step_once(state: DemoState):
    if state.done:
        table = _table_from_observation(state.observation)
        return state, table, "Episode already finished. Click Reset.", _decisions_text(state.decisions), _reward_plot(state.reward_trace)

    agent = HeuristicAgent(seed=42 + len(state.decisions))
    action = agent.choose_action(state.observation)
    next_obs, reward, done, info = state.env.step(action)

    state.observation = next_obs
    state.done = done
    state.total_reward += reward
    state.reward_trace.append(reward)
    state.decisions.append({"action": action, "reward": round(reward, 2), "info": info})

    table = _table_from_observation(state.observation)
    status = (
        f"step={info.get('step')} reward={reward:.2f} total_reward={state.total_reward:.2f} "
        f"done={done} accuracy={info.get('terminal_accuracy', 'n/a')}"
    )
    decisions = _decisions_text(state.decisions)

    return state, table, status, decisions, _reward_plot(state.reward_trace)


def autoplay(state: DemoState, max_steps: int):
    for _ in range(max_steps):
        if state.done:
            break
        state, _, _, _, _ = step_once(state)

    table = _table_from_observation(state.observation)
    final_info = state.decisions[-1]["info"] if state.decisions else {}
    status = (
        f"Autoplay done. steps={len(state.decisions)} total_reward={state.total_reward:.2f} "
        f"terminal_accuracy={final_info.get('terminal_accuracy', 'n/a')} "
        f"vulnerabilities={final_info.get('terminal_vulnerabilities_detected', 'n/a')}"
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
    summary_path = Path("training_outputs/training_summary.json")
    if not summary_path.exists():
        return "No training summary found yet. Run training/train_trl.py first."

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return json.dumps(payload, indent=2)


def comparison_plot_path() -> str | None:
    path = Path("training_outputs/baseline_comparison.png")
    return str(path) if path.exists() else None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ZombieShieldEnv Demo") as demo:
        gr.Markdown(
            "# ZombieShieldEnv\n"
            "Interactive demo for zombie API discovery, classification, and mitigation.\n\n"
            "Use this UI to show what the agent observes, what action it takes each step, and how reward evolves."
        )

        state = gr.State(init_demo())

        with gr.Row():
            seed = gr.Number(value=42, label="Seed", precision=0)
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

        reset_btn.click(reset_env, inputs=[seed], outputs=[state, api_table, status, decisions, reward_plot])
        step_btn.click(step_once, inputs=[state], outputs=[state, api_table, status, decisions, reward_plot])
        auto_btn.click(autoplay, inputs=[state, auto_steps], outputs=[state, api_table, status, decisions, reward_plot])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
