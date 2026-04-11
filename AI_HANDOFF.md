# DevOps Release Commander — AI Handoff Context
*Updated: 2026-04-09 (Final Compliance Phase)*

Welcome, Assistant. If you are reading this, you are taking over or analysing the **DevOps Release Commander** repository for the Meta PyTorch OpenEnv Hackathon. 

## 1. Project Goal & Current State
**Status:** PRODUCTION-READY & COMPLIANT. All known validator parsing and range issues have been resolved.
**Objective:** A fully compliant Reinforcement Learning environment built to the `OpenEnv` specification. It tasks an LLM with acting as a Senior Release Engineer through a 4-stage CI/CD pipeline (PR Triage, Build Gate, Deployment, Monitoring).

Live on both GitHub (`nitishsingh10/Devops-MetaPytorch`) and HuggingFace Spaces (`rolex10/DevOps-Release-Commander`).

## 2. Core Architecture & Critical Design Decisions

### `environment.py` (The Mathematical Core)
- **Universal Safety Clamp:** `_generate_observation()` enforces openenv.yaml bounds on ALL fields (e.g., `0.01` to `99.99`).
- **Reward Clamping:** `_compute_final_reward()` returns `round(max(0.01, min(0.99, normalised)), 2)`. Ensures no `0.0` or `1.0` is ever returned.
- **Deterministic RNG:** Uses `self.rng = random.Random(seed)` internally.
- **Cascading State:** Decisions in early stages (PR/Build) impact Stage 4 metrics (latency, error rates).

### `server/app.py` (The Validator Entry Point)
- **CRITICAL:** `pyproject.toml` declares `server = "server.app:main"`.
- Implements a standalone grading harness with a deterministic baseline policy.
- Uses **exact task names** from `openenv.yaml` (e.g., "PR Triage").
- All tasks reset with `seed=42` for parity with `inference.py`.
- Emits standardized logs: `[START] task="Task Name"`, `[STEP] ...`, `[END] task="Task Name" score=0.XXXX`.

### `inference.py` (The LLM Evaluation Runner)
- Iterates through 5 Tasks using `seed=42`.
- Standardized logging format identical to `server/app.py`.
- **Safety:** Import-time crashes avoided by moving HF/API key checks into `main()`.
- **Clamping:** Explicitly clamps rewards before printing in `[STEP]` and `[END]` logs.

### `openenv.yaml` (The Schema)
- `spec_version: 1`, `reward_range: [0.01, 0.99]`
- 5 tasks with matching `score_range`.
- Graders: `deterministic_pipeline` and `deterministic_pipeline_with_recovery`.

---

## 3. Recently Fixed Bugs (Final Stabilization)

| # | Bug | Root Cause | Fix |
|---|---|---|---|
| 7 | **Invalid [END] Format** | Missing `task=` and `score=` fields in logs. | Updated both scripts to emit `[END] task="..." score=0.XXXX`. |
| 8 | **Task Name Mismatch** | Used safe/slugified names (e.g. `pr_triage`). | Switched to exact names from `openenv.yaml` (e.g. `PR Triage`). |
| 9 | **Seed Inconsistency** | `server/app.py` used sequential seeds (42, 43...). | Aligned all tasks in both scripts to use `seed=42`. |
| 10 | **Module-level Crash** | `raise ValueError` if HF_TOKEN missing on import. | Moved check inside `main()`; prints error instead of crashing. |
| 11 | **Reward Leakage** | `[STEP]` log printed raw un-clamped rewards. | Added explicit `safe_reward` clamping before every print. |

---

## 4. Important Constraints — Read Before Editing

- **Do NOT break the logging syntax.** The validator parses `[START]`, `[STEP]`, `[END]` with strict regex.
- **Do NOT return `0.0` or `1.0`.** Rewards and scores must be strictly in `[0.01, 0.99]`.
- **Do NOT mismatch Task Names.** Task strings in `inference.py` and `server/app.py` must match `openenv.yaml` exactly.
- **Always maintain `seed=42`.** This is our "gold standard" for reproducibility.
- **Zero leakage.** `info` dict must always be `{}` in `env.step()`.

*The repository is currently in its most stable and compliant state.*
