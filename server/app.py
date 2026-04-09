"""
DevOps Release Commander — OpenEnv Server Entry Point
=====================================================
This serves the environment through the openenv-core framework,
which the hackathon validator uses to evaluate task scores.
"""

import json
import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import DevOpsReleaseCmdEnv

# ── Standalone grading harness ─────────────────────────────
# The validator may call `server.app:main` directly.
# This runs all 5 tasks, prints [START]/[STEP]/[END] logs,
# and ensures every reward is strictly in (0, 1).

def main():
    """Run all 5 tasks and emit OpenEnv-compliant scoring logs."""
    env = DevOpsReleaseCmdEnv()

    tasks = [
        {"name": "PR Triage",          "difficulty": 1, "seed": 42},
        {"name": "Build Gate",         "difficulty": 2, "seed": 43},
        {"name": "Full Release",       "difficulty": 3, "seed": 44},
        {"name": "Nightmare Release",  "difficulty": 4, "seed": 45},
        {"name": "Deceptive Release",  "difficulty": 4, "seed": 46},
    ]

    for task in tasks:
        task_name = task["name"]
        safe_name = task_name.replace(" ", "_").lower()
        difficulty = task["difficulty"]
        seed = task["seed"]

        obs_str = env.reset(difficulty=difficulty, seed=seed)
        print(f"[START] task={task_name} env=devops_release_commander", flush=True)

        step_num = 0
        done = False
        step_rewards = []

        # Deterministic baseline policy: always approve (action=0)
        # unless there's a P1 alert (action=1) or high traffic (action=2)
        while not done and step_num < 6:
            step_num += 1
            obs = json.loads(obs_str)

            # Simple rule-based policy
            action = _baseline_policy(obs)

            obs_str, reward, done, info = env.step(action)

            # Clamp reward to strict (0, 1)
            reward = max(0.01, min(0.99, float(reward)))
            step_rewards.append(reward)

            done_val = str(done).lower()
            print(f"[STEP] step={step_num} action={action} reward={reward:.2f} done={done_val} error=null", flush=True)

        # Final score
        final_score = step_rewards[-1] if step_rewards else 0.01
        final_score = max(0.01, min(0.99, final_score))

        success_val = "true" if step_rewards and step_rewards[-1] >= 0.5 else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.01"
        print(f"[END] task={task_name} score={final_score:.4f}", flush=True)


def _baseline_policy(obs: dict) -> int:
    """
    Simple deterministic baseline policy.
    Returns action 0-3 based on observation heuristics.
    """
    stage = obs.get("stage", 1)
    alerts = obs.get("active_alerts", [])
    has_p1 = any("P1" in str(a) for a in alerts)

    # Stage 4/4b: P1 alerts → rollback
    if stage in (4, "4b") and has_p1:
        return 1  # Rollback

    # Stage 4/4b: non-P1 alerts → reduce traffic
    if stage in (4, "4b") and alerts:
        return 2  # Reduce traffic

    # Stage 3: high traffic → canary
    traffic = obs.get("traffic_level_pct", 0)
    if stage == 3 and traffic > 85:
        return 2  # Canary deploy

    # Stage 2: failing build → block
    if stage == 2 and obs.get("build_status") == "failing":
        return 1  # Block

    # Stage 2: flaky build → request changes
    if stage == 2 and obs.get("build_status") == "flaky":
        return 2  # Request changes

    # Stage 1: security risks → block
    diff = obs.get("pr_diff_summary", "")
    security_patterns = ["SQL", "injection", "DROP", "eval(", "exec(", "token =", "exfiltration"]
    if stage == 1 and any(p.lower() in diff.lower() for p in security_patterns):
        return 1  # Block

    # Default: approve
    return 0


if __name__ == "__main__":
    main()
