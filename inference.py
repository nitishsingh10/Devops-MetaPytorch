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

# ── Strict env var loading ─────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
    raise ValueError(
        "Missing required environment variables. "
        "Set: API_BASE_URL, MODEL_NAME, and HF_TOKEN (or API_KEY)"
    )


# ── Per-call timeout handler ───────────────────────────────
def timeout_handler(signum, frame):
    raise TimeoutError("LLM call exceeded 45 second limit")


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
    """Call LLM with 45-second timeout. Returns action int."""
    signal.alarm(45)
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
        print(f"    Raw output: {raw}")

        # Robust parsing — try JSON first, fall back to regex
        try:
            parsed = json.loads(raw)
            action = int(parsed.get("action", -1))
            reason = parsed.get("reason", "")
            if reason:
                print(f"    Reason: {reason}")
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\b([0-3])\b", raw)
            action = int(match.group(1)) if match else -1

        return action if action in (0, 1, 2, 3) else -1

    except TimeoutError:
        signal.alarm(0)
        print("    LLM TIMEOUT — using fallback action -1")
        return -1
    except Exception as e:
        signal.alarm(0)
        print(f"    LLM ERROR: {e} — using fallback action -1")
        return -1


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = DevOpsReleaseCmdEnv()
    difficulties = [1, 2, 3, 4]
    task_names = [
        "Easy PR Triage",
        "Build Gate",
        "Full Release",
        "Nightmare Release",
    ]
    results = []

    print("=" * 60)
    print("  DevOps Release Commander — Agent Evaluation")
    print("=" * 60)

    start_total = time.time()

    for i, diff in enumerate(difficulties):
        task_id = i + 1
        task_name = task_names[i]

        print(f"\n{'─' * 60}")
        print(f"Task {task_id}: {task_name} (Difficulty {diff})")
        print(f"{'─' * 60}")

        obs_str = env.reset(difficulty=diff, seed=42)
        episode_reward = 0.0
        step_num = 0
        done = False

        # ── [START] structured log ────────────────────────────
        print(f'[START] task={task_id} difficulty={diff} name="{task_name}"')

        while not done:
            step_num += 1
            obs = json.loads(obs_str)
            stage = obs.get("stage", "?")
            print(f"\n  Stage {stage} (Step {step_num}):")
            print(
                f"  Obs summary: trust={obs.get('author_trust_score', 'N/A')} "
                f"alerts={obs.get('active_alerts', [])} "
                f"build={obs.get('build_status', 'N/A')}"
            )

            action = call_llm(client, obs_str)
            print(f"  Action: {action}")

            obs_str, reward, done, info = env.step(action)
            episode_reward = reward
            print(f"  Step reward: {reward} | done: {done}")

            # ── [STEP] structured log ─────────────────────────
            print(f"[STEP] task={task_id} step={step_num} stage={stage} action={action} reward={reward} done={done}")

            if step_num > 6:  # Safety guard against infinite loops
                done = True

        # ── Determine status ──────────────────────────────────
        if episode_reward == 1.0:
            status = "optimal"
            print(f"\n  EPISODE REWARD: {episode_reward}")
            print("  STATUS: Optimal ✅")
        elif episode_reward == 0.0:
            status = "catastrophic"
            print(f"\n  EPISODE REWARD: {episode_reward}")
            print("  STATUS: Catastrophic ❌")
        else:
            status = "partial"
            print(f"\n  EPISODE REWARD: {episode_reward}")
            print(f"  STATUS: Partial {episode_reward} ⚠️")

        # ── [END] structured log ──────────────────────────────
        print(f'[END] task={task_id} reward={episode_reward} status={status} steps={step_num}')

        results.append(
            {"task": task_name, "difficulty": diff, "reward": episode_reward}
        )

    elapsed = time.time() - start_total

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  Task {r['difficulty']} ({r['task']}): {r['reward']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

