"""Minimal FastAPI server wrapper for ZombieShieldEnv local/remote execution."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.zombieshield_env import ZombieShieldEnv


app = FastAPI(title="ZombieShieldEnv API", version="0.1.0")
env = ZombieShieldEnv(seed=42)
current_state: Dict[str, Any] | None = None


class StepRequest(BaseModel):
    action: Any


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset() -> Dict[str, Any]:
    global current_state
    current_state = env.reset()
    return {"state": current_state}


@app.get("/state")
def state() -> Dict[str, Any]:
    if current_state is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    return {"state": env.state()}


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    global current_state
    if current_state is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")

    next_state, reward, done, info = env.step(payload.action)
    current_state = next_state
    return {
        "state": next_state,
        "reward": reward,
        "done": done,
        "info": info,
    }
