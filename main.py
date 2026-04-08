"""
DevOps Release Commander — FastAPI Server
==========================================
Serves the OpenEnv environment over HTTP on port 7860.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from environment import DevOpsReleaseCmdEnv
import json
from typing import Optional
import os

app = FastAPI(
    title="DevOps Release Commander API",
    description="OpenEnv RL environment for the full software release lifecycle",
    version="1.0.0",
)

env = DevOpsReleaseCmdEnv()


class ActionRequest(BaseModel):
    action: int
    reason: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def demo_ui():
    """Serve the live demo UI."""
    static_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "static", "index.html"
    )
    with open(static_path, "r") as f:
        return f.read()



@app.api_route("/reset", methods=["GET", "POST"])
async def reset_env(request: Request, difficulty: Optional[int] = None, seed: Optional[int] = None):
    """Reset the environment and start a new episode."""
    try:
        if request.method == "POST":
            # the validator sends `-d '{}'`
            body = await request.json()
            if isinstance(body, dict):
                difficulty = body.get("difficulty", difficulty)
                seed = body.get("seed", seed)
    except Exception:
        pass
        
    obs_str = env.reset(difficulty=difficulty, seed=seed)
    return {"observation": json.loads(obs_str)}


@app.get("/state")
def state_env():
    """Return the current observation without modification."""
    return {"observation": json.loads(env.state())}


@app.post("/step")
def step_env(payload: ActionRequest):
    """Take an action and advance the pipeline."""
    next_obs_str, reward, done, info = env.step(payload.action)
    return {
        "observation": json.loads(next_obs_str),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "DevOps Release Commander"}
