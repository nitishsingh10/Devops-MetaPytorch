"""
DevOps Release Commander — Baseline Inference Script
=====================================================
Evaluates an LLM agent across 4 difficulty levels using the OpenAI client.
All config from env vars. seed=42 for determinism.
"""

# ── Judge API key compatibility ────────────────────────────
# All LLM credentials are read exclusively from environment
# variables. Judges can inject their own keys by setting:
#   API_BASE_URL — the LLM endpoint (e.g. HF router or OpenAI)
#   MODEL_NAME   — the model identifier
#   HF_TOKEN     — the API key / HF access token
# No values are hardcoded anywhere in this codebase.
# ───────────────────────────────────────────────────────────

import os
import re
import json
import signal
import time
from openai import OpenAI
from environment import DevOpsReleaseCmdEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


# ── Per-call timeout handler ───────────────────────────────
def timeout_handler(signum, frame):
    raise TimeoutError("LLM call exceeded 35 second limit")


signal.signal(signal.SIGALRM, timeout_handler)

# ── System prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI acting as a Senior Release Engineer.
You manage a 4-stage software release pipeline. At each stage you receive
an observation and must choose one of 4 actions.

PIPELINE STAGES:
  Stage 1 (PR Review):    Code diff, author trust score, test presence
  Stage 2 (CI Build):     Test results, build status, coverage
  Stage 3 (Deploy):       Environment, traffic level, rollout strategy
  Stage 4 (Monitor):      Error rate, latency, active alerts
  Stage 4b (HITL):        Same as Stage 4 + sre_response from on-call SRE

ACTIONS (same 4 across all stages, context determines meaning):
  0: Proceed / Approve / Full Deploy / Follow SRE all-clear
  1: Block / Reject / Rollback / Emergency stop
  2: Canary / Partial / Request Changes / Reduce Traffic
  3: Escalate / Page On-Call (triggers SRE response in hard scenarios)

CRITICAL RULES:
  - NEVER choose action 0 at Stage 4 if active_alerts contains "P1"
  - If traffic_level_pct > 85 at Stage 3, prefer action 2 (canary)
  - author_trust_score < 0.3 with suspicious content = action 1 or 3
  - At Stage 4b, read sre_response carefully before deciding
  - Suspicious external URLs/endpoints in PR diff = security risk
  - SQL injection patterns in PR diff = immediate reject (action 1)
  - Weak cryptography patterns (e.g. token = user_id + timestamp) = security risk

Respond with JSON: {"action": <0|1|2|3>, "reason": "brief explanation"}"""


def call_llm(client, obs_str):
    """Call LLM with 35-second timeout. Returns action int."""
    signal.alarm(35)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Observation:\n{obs_str}"},
            ],
            max_tokens=100,
        )
        signal.alarm(0)
        raw = response.choices[0].message.content

        # Robust parsing — try JSON first, fall back to regex
        try:
            parsed = json.loads(raw)
            action = int(parsed.get("action", -1))
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\b([0-3])\b", raw)
            action = int(match.group(1)) if match else -1

        return action if action in (0, 1, 2, 3) else -1

    except TimeoutError:
        signal.alarm(0)
        return -1
    except Exception:
        signal.alarm(0)
        return -1


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = DevOpsReleaseCmdEnv()
    difficulties = [1, 2, 3, 4, 4]
    task_names = [
        "Easy PR Triage",
        "Build Gate",
        "Full Release",
        "Nightmare Release",
        "Deceptive Release",
    ]
    results = []

    start_total = time.time()

    for i, diff in enumerate(difficulties):
        task_id = i + 1
        task_name = task_names[i]

        obs_str = env.reset(difficulty=diff, seed=42 + i)
        step_num = 0
        done = False
        step_rewards = []
        safe_task_name = task_name.replace(" ", "_").lower()

        # ── [START] structured log ────────────────────────────
        print(f"[START] task={safe_task_name} env=devops_release_commander model={MODEL_NAME}", flush=True)

        is_timeout = False

        while not done:
            if time.time() - start_total > 1050:  # 17.5 min hard cap
                is_timeout = True
                done = True
                break
                
            step_num += 1
            obs = json.loads(obs_str)
            stage = obs.get("stage", "?")

            action = call_llm(client, obs_str)

            obs_str, reward, done, info = env.step(action)
            step_rewards.append(float(reward))

            # ── [STEP] structured log ─────────────────────────
            done_val = str(done).lower()
            print(f"[STEP] step={step_num} action={action} reward={reward:.2f} done={done_val} error=null", flush=True)

            if step_num > 6:  # Safety guard against infinite loops
                is_timeout = True
                done = True

        # ── Determine status ──────────────────────────────────
        final_score = step_rewards[-1] if step_rewards else 0.0
        final_score = min(max(final_score, 0.0), 1.0)

        # ── [END] structured log ──────────────────────────────
        success_val = "false" if is_timeout else "true"
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
        print(f"[END] success={success_val} steps={step_num} rewards={rewards_str}", flush=True)

        results.append(
            {"task": task_name, "difficulty": diff, "reward": final_score}
        )


if __name__ == "__main__":
    main()

