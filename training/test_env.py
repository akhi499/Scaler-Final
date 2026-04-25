"""Minimal local smoke tests for ZombieShieldEnv."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.zombieshield_env import ZombieShieldEnv


def main() -> None:
    env = ZombieShieldEnv(min_apis=10, max_apis=12, seed=123)
    obs = env.reset()
    print("reset_ok visible_api_count=", obs["visible_api_count"])

    api_id = obs["apis"][0]["api_id"] if obs.get("apis") else "api_0"
    actions = [
        f"scan_api({api_id})",
        f"run_security_test({api_id})",
        f"classify_api({api_id},ZOMBIE)",
        f"ignore_api({api_id})",
    ]

    for i, action in enumerate(actions, start=1):
        obs, reward, done, info = env.step(action)
        print(f"step_{i} action={action} reward={reward:.2f} done={done} status={info.get('status')}")

    # Multi-episode stability check.
    stable = True
    for ep in range(5):
        env.reset()
        for _ in range(6):
            apis = env.state().get("apis", [])
            aid = apis[0]["api_id"] if apis else "api_0"
            _, _, done, _ = env.step(f"scan_api({aid})")
            if done:
                break
        if ep == 4 and done:
            pass

    print("multi_episode_ok", stable)


if __name__ == "__main__":
    main()
