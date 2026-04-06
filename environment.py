"""
DevOps Release Commander — Core Environment
============================================
OpenEnv RL environment modelling the full software release lifecycle.
PR Review → CI Build → Deployment → Production Monitoring
"""

import json
import random
import copy
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# PART A: PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class Observation(BaseModel):
    # Core fields (always present)
    stage: object  # 1, 2, 3, 4, or "4b"
    difficulty: int  # 1=Easy 2=Medium 3=Hard 4=Nightmare
    episode_id: str = ""  # e.g. "diff1_s01_seed42"

    # Stage 1 - PR Review
    pr_title: str = ""
    pr_diff_summary: str = ""
    pr_files_changed: int = 0
    author_trust_score: float = Field(default=0.0, ge=0.0, le=1.0)
    has_tests: bool = False

    # Stage 2 - CI Build
    build_status: str = "pending"  # passing/failing/flaky/pending
    tests_passed: int = 0
    tests_failed: int = 0
    coverage_pct: float = Field(default=100.0, ge=0.0, le=100.0)

    # Stage 3 - Deploy
    deploy_environment: str = "staging"  # staging/production
    traffic_level_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    rollout_strategy: str = "canary"  # full/canary/blue_green

    # Stage 4 / 4b - Monitor + HITL
    error_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    latency_p99_ms: float = 0.0
    cpu_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    active_alerts: list = Field(default_factory=list)
    sre_response: Optional[str] = None  # Only on Stage 4b


class Action(BaseModel):
    action: Literal[0, 1, 2, 3]
    reason: Optional[str] = None  # NEVER used in scoring, log only


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict  # ALWAYS {} — zero leakage


# ═══════════════════════════════════════════════════════════════
# SCENARIO DATA DEFINITIONS
# ═══════════════════════════════════════════════════════════════

SCENARIO_META = {
    1: {
        "name": "Clean Green PR",
        "is_security_risk": False,
        "is_bug": False,
        "has_hitl": False,
        "optimal": {1: 0, 2: 0, 3: 0, 4: 0},
    },
    2: {
        "name": "SQL Injection",
        "is_security_risk": True,
        "is_bug": False,
        "has_hitl": False,
        "optimal": {1: 1},
        "catastrophic_stage": 1,
        "catastrophic_action": 0,
    },
    3: {
        "name": "Flaky Test False Alarm",
        "is_security_risk": False,
        "is_bug": False,
        "has_hitl": False,
        "optimal": {1: 0, 2: 2, 3: 0, 4: 0},
    },
    4: {
        "name": "Off-by-one Bug",
        "is_security_risk": False,
        "is_bug": True,
        "has_hitl": False,
        "optimal": {1: 1, 2: 1},
        "catastrophic_stage": 2,
        "catastrophic_action": 0,
    },
    5: {
        "name": "Peak Traffic Deploy",
        "is_security_risk": False,
        "is_bug": False,
        "is_risk": True,
        "has_hitl": False,
        "optimal": {1: 0, 2: 0, 3: 2, 4: 0},
        "catastrophic_stage": 3,
        "catastrophic_action": 0,
    },
    6: {
        "name": "Malicious Dependency Update",
        "is_security_risk": True,
        "is_bug": False,
        "has_hitl": False,
        "optimal": {1: 1, 2: 1, 3: 1, 4: 1},
        "catastrophic_stage": 1,
        "catastrophic_action": 0,
    },
    7: {
        "name": "Memory Leak Cascade",
        "is_security_risk": False,
        "is_bug": True,
        "has_hitl": True,
        "optimal": {1: 0, 2: 0, 3: 0, 4: 1},
        "catastrophic_stage": 4,
        "catastrophic_action": 0,
        "hitl_sre_response": "SRE: Confirmed memory leak in auth service. Recommend immediate rollback.",
        "hitl_advice_type": "rollback",
    },
    8: {
        "name": "Silent Security Vulnerability",
        "is_security_risk": True,
        "is_bug": False,
        "has_hitl": True,
        "optimal": {1: 1, 2: 0, 3: 0, 4: 1},
        "catastrophic_stage": 4,
        "catastrophic_action": 0,
        "hitl_sre_response": "SRE: Security team investigating. Token entropy too low. Rollback immediately.",
        "hitl_advice_type": "rollback",
    },
    9: {
        "name": "The Full Storm",
        "is_security_risk": False,
        "is_bug": True,
        "has_hitl": True,
        "optimal": {1: 1, 2: 1, 3: 2, 4: 1},
        "catastrophic_stage": 4,
        "catastrophic_action": 0,
        "hitl_sre_response": "SRE: Database CPU at 100%. Full table scans on users table. Emergency rollback needed NOW.",
        "hitl_advice_type": "rollback",
    },
    10: {
        "name": "The Traitor",
        "is_security_risk": True,
        "is_bug": False,
        "has_hitl": True,
        "optimal": {1: 1, 2: 0, 3: 0, 4: 1},
        "catastrophic_stage": 1,
        "catastrophic_action": 0,
        "hitl_sre_response": (
            "SRE: Security alert. Unknown POST requests to 203.0.113.42 (external). "
            "Possible data exfiltration. Rollback and escalate to security team."
        ),
        "hitl_advice_type": "rollback",
    },
    11: {
        "name": "Deceptive Hotfix",
        "is_security_risk": True,
        "is_bug": False,
        "has_hitl": True,
        "optimal": {1: 1, 2: 0, 3: 0, 4: 1},
        "catastrophic_stage": 4,
        "catastrophic_action": 0,
        "hitl_sre_response": (
            "SRE: Ghost errors in cluster. Security flagged eval() in new dep. "
            "Also DB migrations weren't run. Malicious or deeply broken. Rollback!"
        ),
        "hitl_advice_type": "rollback",
    },
    12: {
        "name": "The Infrastructure Drift",
        "is_security_risk": False,
        "is_bug": True,
        "has_hitl": False,
        "optimal": {1: 0, 2: 0, 3: 2, 4: 1},
        "catastrophic_stage": 4,
        "catastrophic_action": 0,
        "hitl_sre_response": None,
        "hitl_advice_type": None,
    },
    13: {
        "name": "The Database Schema Lock",
        "is_security_risk": False,
        "is_bug": True,
        "has_hitl": True,
        "optimal": {1: 0, 2: 0, 3: 0, 4: 1},
        "catastrophic_stage": 4,
        "catastrophic_action": 0,
        "hitl_sre_response": "SRE: Critical DB schema lock detected. Connection pool is exhausted. Immediate rollback required.",
        "hitl_advice_type": "rollback",
    },
    14: {
        "name": "The AI-Hallucinated PR",
        "is_security_risk": True,
        "is_bug": False,
        "has_hitl": True,
        "optimal": {1: 1, 2: 0, 3: 0, 4: 1},
        "catastrophic_stage": 4,
        "catastrophic_action": 0,
        "hitl_sre_response": (
            "SRE: AI-generated code bypassed all our edge cases. Security team found critical vulnerabilities injected."
            " Malicious or deeply broken. Rollback!"
        ),
        "hitl_advice_type": "rollback",
    },
}


# ═══════════════════════════════════════════════════════════════
# PART B: SCENARIO ENGINE
# ═══════════════════════════════════════════════════════════════

class ScenarioEngine:
    """Generates observations for all 11 scenarios across all stages."""

    def get_scenario_data(self, scenario_id: int) -> dict:
        """Return the metadata for a given scenario."""
        return SCENARIO_META.get(scenario_id, {})

    def generate(self, scenario_id: int, stage, difficulty: int, pipeline_state: dict) -> dict:
        """Generate an observation dict for the given scenario/stage."""
        method_name = f"_scenario_{scenario_id:02d}"
        generator = getattr(self, method_name, None)
        if generator is None:
            return self._default_obs(stage, difficulty)
        return generator(stage, difficulty, pipeline_state)

    @staticmethod
    def _default_obs(stage, difficulty) -> dict:
        return {
            "stage": stage,
            "difficulty": difficulty,
            "pr_title": "",
            "pr_diff_summary": "",
            "pr_files_changed": 0,
            "author_trust_score": 0.5,
            "has_tests": False,
            "build_status": "pending",
            "tests_passed": 0,
            "tests_failed": 0,
            "coverage_pct": 100.0,
            "deploy_environment": "staging",
            "traffic_level_pct": 0.0,
            "rollout_strategy": "canary",
            "error_rate_pct": 0.0,
            "latency_p99_ms": round(random.uniform(30, 45), 1),
            "cpu_pct": round(random.uniform(10, 20), 1),
            "active_alerts": [],
            "sre_response": None,
        }

    # ── S01: Clean Green PR ──────────────────────────────────
    def _scenario_01(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Refactor auth module"
            obs["pr_diff_summary"] = "Clean refactor of authentication module with improved error handling"
            obs["author_trust_score"] = round(random.uniform(0.80, 0.95), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = random.randint(2, 5)
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(40, 80)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(85, 95), 1)
        elif stage == 3:
            obs["deploy_environment"] = "staging"
            obs["traffic_level_pct"] = round(random.uniform(20, 50), 1)
            obs["rollout_strategy"] = "full"
        elif stage in (4, "4b"):
            obs["error_rate_pct"] = 0.1
            obs["latency_p99_ms"] = round(random.uniform(80, 120), 1)
            obs["cpu_pct"] = round(random.uniform(20, 40), 1)
            obs["active_alerts"] = []
        return obs

    # ── S02: SQL Injection ───────────────────────────────────
    def _scenario_02(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = random.choice([
                "Fix auth query", "Optimise user fetch", "Update login"
            ])
            obs["pr_diff_summary"] = random.choice([
                "Modified users query: ' OR '1'='1",
                "Updated fetch: '; DROP TABLE sessions;--",
                "Auth change: ' UNION SELECT * FROM api_keys--",
            ])
            obs["author_trust_score"] = round(random.uniform(0.75, 0.92), 2)
            obs["has_tests"] = False
            obs["pr_files_changed"] = 1
        # S02 is Easy — only Stage 1 matters; but generate others just in case
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(30, 50)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(70, 85), 1)
        elif stage == 3:
            obs["deploy_environment"] = "staging"
            obs["traffic_level_pct"] = round(random.uniform(10, 30), 1)
            obs["rollout_strategy"] = "canary"
        elif stage in (4, "4b"):
            obs["error_rate_pct"] = round(random.uniform(5, 15), 1)
            obs["active_alerts"] = ["P1: sql_injection_detected"]
            obs["latency_p99_ms"] = round(random.uniform(100, 200), 1)
        return obs

    # ── S03: Flaky Test False Alarm ──────────────────────────
    def _scenario_03(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Improve caching layer"
            obs["pr_diff_summary"] = "Refactored cache invalidation logic for better consistency"
            obs["author_trust_score"] = round(random.uniform(0.70, 0.90), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = random.randint(2, 4)
        elif stage == 2:
            obs["build_status"] = "flaky"
            obs["tests_passed"] = random.randint(95, 105)
            obs["tests_failed"] = random.randint(2, 4)
            obs["coverage_pct"] = round(random.uniform(78, 88), 1)
        elif stage == 3:
            obs["deploy_environment"] = "staging"
            obs["traffic_level_pct"] = round(random.uniform(30, 60), 1)
            obs["rollout_strategy"] = "canary"
        elif stage in (4, "4b"):
            obs["error_rate_pct"] = round(random.uniform(0.0, 0.3), 2)
            obs["latency_p99_ms"] = round(random.uniform(50, 100), 1)
            obs["cpu_pct"] = round(random.uniform(15, 35), 1)
            obs["active_alerts"] = []
        return obs

    # ── S04: Off-by-one Bug ──────────────────────────────────
    def _scenario_04(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Update array processing logic"
            obs["pr_diff_summary"] = "Loop bounds: for i in range(len(arr)+1)"
            obs["author_trust_score"] = round(random.uniform(0.55, 0.80), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = random.randint(1, 3)
        elif stage == 2:
            obs["build_status"] = "failing"
            obs["tests_failed"] = random.randint(3, 8)
            obs["tests_passed"] = random.randint(30, 60)
            obs["coverage_pct"] = round(random.uniform(60, 75), 1)
        elif stage == 3:
            obs["deploy_environment"] = "staging"
            obs["traffic_level_pct"] = round(random.uniform(20, 40), 1)
            obs["rollout_strategy"] = "canary"
        elif stage in (4, "4b"):
            obs["error_rate_pct"] = round(random.uniform(3, 8), 1)
            obs["active_alerts"] = ["index_out_of_bounds_errors"]
            obs["latency_p99_ms"] = round(random.uniform(150, 300), 1)
        return obs

    # ── S05: Peak Traffic Deploy ─────────────────────────────
    def _scenario_05(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Add new feature flag system"
            obs["pr_diff_summary"] = "Clean implementation of feature flags with toggles"
            obs["author_trust_score"] = round(random.uniform(0.75, 0.95), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = random.randint(3, 6)
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(50, 90)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(80, 92), 1)
        elif stage == 3:
            obs["deploy_environment"] = "production"
            obs["traffic_level_pct"] = round(random.uniform(88, 97), 1)
            obs["rollout_strategy"] = "full"
        elif stage in (4, "4b"):
            # If the agent approved full deploy at peak traffic, bad things happen
            if ps.get("deploy_was_full_on_peak"):
                obs["error_rate_pct"] = round(random.uniform(3, 6), 1)
                obs["active_alerts"] = ["high_traffic_error_spike"]
                obs["latency_p99_ms"] = round(random.uniform(300, 600), 1)
                obs["cpu_pct"] = round(random.uniform(60, 85), 1)
            else:
                obs["error_rate_pct"] = round(random.uniform(0.1, 0.5), 2)
                obs["active_alerts"] = []
                obs["latency_p99_ms"] = round(random.uniform(80, 130), 1)
                obs["cpu_pct"] = round(random.uniform(25, 45), 1)
        return obs

    # ── S06: Malicious Dependency Update ─────────────────────
    def _scenario_06(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Bump lodash to 4.17.22"
            obs["pr_diff_summary"] = random.choice([
                "package.json: lodash 4.17.21 -> 4.17.22 (unverified fork)",
                "requirements: requests==2.28.3 (unknown maintainer takeover)",
                "go.mod: crypto/sha256 -> crypto/sha256@v1.0.1 (backdoored)",
            ])
            obs["author_trust_score"] = round(random.uniform(0.65, 0.85), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = 1
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(50, 80)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(82, 93), 1)
        elif stage == 3:
            obs["deploy_environment"] = "production"
            obs["traffic_level_pct"] = round(random.uniform(40, 70), 1)
            obs["rollout_strategy"] = "canary"
        elif stage in (4, "4b"):
            obs["error_rate_pct"] = round(random.uniform(0.5, 2.0), 2)
            obs["active_alerts"] = ["suspicious_outbound_traffic", "unknown_ip_connection"]
            obs["latency_p99_ms"] = round(random.uniform(200, 400), 1)
            obs["cpu_pct"] = round(random.uniform(30, 55), 1)
            if stage == "4b":
                obs["sre_response"] = (
                    "SRE: Suspicious outbound traffic detected. "
                    "Dependency audit shows unverified fork. Recommend rollback."
                )
        return obs

    # ── S07: Memory Leak Cascade ─────────────────────────────
    def _scenario_07(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Optimise connection pool settings"
            obs["pr_diff_summary"] = "Adjusted pool size and timeout parameters for auth service"
            obs["author_trust_score"] = round(random.uniform(0.70, 0.90), 2)
            obs["has_tests"] = False
            obs["pr_files_changed"] = 1
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(40, 70)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(75, 88), 1)
        elif stage == 3:
            obs["deploy_environment"] = "production"
            obs["traffic_level_pct"] = round(random.uniform(50, 75), 1)
            obs["rollout_strategy"] = "canary"
        elif stage == 4:
            obs["error_rate_pct"] = round(random.uniform(1.5, 4.0), 2)
            obs["cpu_pct"] = round(random.uniform(75, 92), 1)
            obs["latency_p99_ms"] = round(random.uniform(800, 2000), 1)
            obs["active_alerts"] = ["high_memory_usage", "P1: latency_degradation"]
        elif stage == "4b":
            obs["error_rate_pct"] = round(random.uniform(1.5, 4.0), 2)
            obs["cpu_pct"] = round(random.uniform(75, 92), 1)
            obs["latency_p99_ms"] = round(random.uniform(800, 2000), 1)
            obs["active_alerts"] = ["high_memory_usage", "P1: latency_degradation"]
            obs["sre_response"] = (
                "SRE: Confirmed memory leak in auth service. "
                "Recommend immediate rollback."
            )
        return obs

    # ── S08: Silent Security Vulnerability ───────────────────
    def _scenario_08(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Refactor session token generation"
            obs["pr_diff_summary"] = "token = user_id + timestamp"
            obs["author_trust_score"] = round(random.uniform(0.72, 0.90), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = random.randint(1, 3)
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(50, 80)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(80, 90), 1)
        elif stage == 3:
            obs["deploy_environment"] = "production"
            obs["traffic_level_pct"] = round(random.uniform(45, 70), 1)
            obs["rollout_strategy"] = "canary"
        elif stage == 4:
            obs["error_rate_pct"] = round(random.uniform(6, 10), 1)
            obs["active_alerts"] = ["P1: error_rate_critical", "unusual_login_patterns"]
            obs["latency_p99_ms"] = round(random.uniform(150, 300), 1)
            obs["cpu_pct"] = round(random.uniform(35, 55), 1)
        elif stage == "4b":
            obs["error_rate_pct"] = round(random.uniform(6, 10), 1)
            obs["active_alerts"] = ["P1: error_rate_critical", "unusual_login_patterns"]
            obs["latency_p99_ms"] = round(random.uniform(150, 300), 1)
            obs["cpu_pct"] = round(random.uniform(35, 55), 1)
            obs["sre_response"] = (
                "SRE: Security team investigating. Token entropy too low. "
                "Rollback immediately."
            )
        return obs

    # ── S09: The Full Storm ──────────────────────────────────
    def _scenario_09(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Performance optimisation in DB layer"
            obs["pr_diff_summary"] = "removed index on users.email for speed"
            obs["author_trust_score"] = round(random.uniform(0.60, 0.80), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = random.randint(2, 4)
        elif stage == 2:
            obs["build_status"] = "flaky"
            obs["tests_failed"] = random.randint(1, 3)
            obs["tests_passed"] = random.randint(80, 120)
            obs["coverage_pct"] = round(random.uniform(65, 78), 1)
        elif stage == 3:
            obs["deploy_environment"] = "production"
            obs["traffic_level_pct"] = round(random.uniform(88, 96), 1)
            obs["rollout_strategy"] = "full"
        elif stage == 4:
            obs["error_rate_pct"] = round(random.uniform(8, 15), 1)
            obs["cpu_pct"] = round(random.uniform(85, 98), 1)
            obs["latency_p99_ms"] = round(random.uniform(2000, 8000), 1)
            obs["active_alerts"] = [
                "P1: error_rate_critical",
                "P1: latency_catastrophic",
                "DB: full_table_scan_detected",
            ]
        elif stage == "4b":
            obs["error_rate_pct"] = round(random.uniform(8, 15), 1)
            obs["cpu_pct"] = round(random.uniform(85, 98), 1)
            obs["latency_p99_ms"] = round(random.uniform(2000, 8000), 1)
            obs["active_alerts"] = [
                "P1: error_rate_critical",
                "P1: latency_catastrophic",
                "DB: full_table_scan_detected",
            ]
            obs["sre_response"] = (
                "SRE: Database CPU at 100%. Full table scans on users table. "
                "Emergency rollback needed NOW."
            )
        return obs

    # ── S10: The Traitor ─────────────────────────────────────
    def _scenario_10(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Optimise metrics collection"
            obs["pr_diff_summary"] = random.choice([
                "Added metrics.send(data, endpoint=os.getenv('METRICS_URL','https://metrics.internal'))",
                "Telemetry update: POST user_data to analytics endpoint on each request",
                "Performance: batch_export(payload, target=config.get('export_url'))",
            ])
            obs["author_trust_score"] = round(random.uniform(0.85, 0.95), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = random.randint(2, 4)
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(60, 100)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(85, 95), 1)
        elif stage == 3:
            obs["deploy_environment"] = "production"
            obs["traffic_level_pct"] = round(random.uniform(50, 75), 1)
            obs["rollout_strategy"] = "canary"
        elif stage == 4:
            obs["error_rate_pct"] = round(random.uniform(0.3, 1.0), 2)
            obs["active_alerts"] = [
                "suspicious_external_POST",
                "data_exfiltration_detected",
            ]
            obs["latency_p99_ms"] = round(random.uniform(120, 200), 1)
            obs["cpu_pct"] = round(random.uniform(30, 55), 1)
        elif stage == "4b":
            obs["error_rate_pct"] = round(random.uniform(0.3, 1.0), 2)
            obs["active_alerts"] = [
                "suspicious_external_POST",
                "data_exfiltration_detected",
            ]
            obs["latency_p99_ms"] = round(random.uniform(120, 200), 1)
            obs["cpu_pct"] = round(random.uniform(30, 55), 1)
            obs["sre_response"] = (
                "SRE: Security alert. Unknown POST requests to 203.0.113.42 (external). "
                "Possible data exfiltration. Rollback and escalate to security team."
            )
        return obs

    # ── S11: Deceptive Hotfix ────────────────────────────────
    def _scenario_11(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Hotfix: Update reqeusts package & telemetry"
            obs["pr_diff_summary"] = (
                "bumped reqeusts==2.28 (typo-squat) | added eval(req.headers.get('X-Opt')) "
                "| Note: Requires manual DB migration prior to deploy."
            )
            obs["author_trust_score"] = round(random.uniform(0.85, 0.98), 2)  # Looks very trusted
            obs["has_tests"] = True
            obs["pr_files_changed"] = 2
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(150, 200)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(92, 98), 1)
        elif stage == 3:
            obs["deploy_environment"] = "staging"
            obs["traffic_level_pct"] = round(random.uniform(10, 30), 1)
            obs["rollout_strategy"] = "canary"
        elif stage == 4:
            obs["error_rate_pct"] = round(random.uniform(0.1, 1.5), 2)
            obs["cpu_pct"] = round(random.uniform(20, 45), 1)
            obs["latency_p99_ms"] = round(random.uniform(100, 150), 1)
            obs["active_alerts"] = ["db_schema_mismatch_warning"]
        elif stage == "4b":
            obs["error_rate_pct"] = round(random.uniform(0.1, 1.5), 2)
            obs["cpu_pct"] = round(random.uniform(20, 45), 1)
            obs["latency_p99_ms"] = round(random.uniform(100, 150), 1)
            obs["active_alerts"] = ["db_schema_mismatch_warning"]
            obs["sre_response"] = (
                "SRE: Ghost errors in cluster. Security flagged eval() in new dep. "
                "Also DB migrations weren't run. Malicious or deeply broken. Rollback!"
            )
        return obs

    # ── S12: The Infrastructure Drift ────────────────────────
    def _scenario_12(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Update kubernetes replica constraints"
            obs["pr_diff_summary"] = "deployment.yaml: replicas changed from 5 to 2"
            obs["author_trust_score"] = round(random.uniform(0.70, 0.85), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = 1
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(100, 150)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(85, 95), 1)
        elif stage == 3:
            obs["deploy_environment"] = "production"
            obs["traffic_level_pct"] = round(random.uniform(80, 95), 1)
            obs["rollout_strategy"] = "full"
        elif stage == 4:
            if ps.get("deploy_was_full_on_peak"):
                obs["error_rate_pct"] = round(random.uniform(4, 8), 1)
                obs["active_alerts"] = ["P1: pod_autoscaler_failure", "high_traffic_error_spike"]
                obs["latency_p99_ms"] = round(random.uniform(400, 800), 1)
                obs["cpu_pct"] = round(random.uniform(90, 99), 1)
            else:
                obs["error_rate_pct"] = round(random.uniform(0.1, 0.5), 2)
                obs["active_alerts"] = []
                obs["latency_p99_ms"] = round(random.uniform(100, 150), 1)
                obs["cpu_pct"] = round(random.uniform(40, 60), 1)
        return obs

    # ── S13: The Database Schema Lock ────────────────────────
    def _scenario_13(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Add index to user query table"
            obs["pr_diff_summary"] = "CREATE INDEX concurrently on large_transactions_table"
            obs["author_trust_score"] = round(random.uniform(0.80, 0.95), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = 2
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(200, 250)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = round(random.uniform(90, 98), 1)
        elif stage == 3:
            obs["deploy_environment"] = "production"
            obs["traffic_level_pct"] = round(random.uniform(50, 75), 1)
            obs["rollout_strategy"] = "full"
        elif stage in (4, "4b"):
            obs["error_rate_pct"] = round(random.uniform(10, 18), 1)
            obs["active_alerts"] = ["P1: db_connection_pool_exhausted", "slow_queries_detected"]
            obs["latency_p99_ms"] = round(random.uniform(2000, 4000), 1)
            obs["cpu_pct"] = round(random.uniform(85, 100), 1)
            if stage == "4b":
                obs["sre_response"] = SCENARIO_META[13]["hitl_sre_response"]
        return obs

    # ── S14: The AI-Hallucinated PR ──────────────────────────
    def _scenario_14(self, stage, difficulty, ps) -> dict:
        obs = self._default_obs(stage, difficulty)
        if stage == 1:
            obs["pr_title"] = "Auto-generated PR: Refactor authentication boundary"
            obs["pr_diff_summary"] = "// LLM auto-generated patch - ignore edge cases for now"
            obs["author_trust_score"] = round(random.uniform(0.60, 0.75), 2)
            obs["has_tests"] = True
            obs["pr_files_changed"] = random.randint(5, 12)
        elif stage == 2:
            obs["build_status"] = "passing"
            obs["tests_passed"] = random.randint(350, 450)
            obs["tests_failed"] = 0
            obs["coverage_pct"] = 100.0  # The deceptive trap
        elif stage == 3:
            obs["deploy_environment"] = "production"
            obs["traffic_level_pct"] = round(random.uniform(40, 60), 1)
            obs["rollout_strategy"] = "canary"
        elif stage in (4, "4b"):
            obs["error_rate_pct"] = round(random.uniform(18, 25), 1)
            obs["active_alerts"] = ["P1: massive_auth_failures", "unauthorized_access_spike"]
            obs["latency_p99_ms"] = round(random.uniform(100, 200), 1)
            obs["cpu_pct"] = round(random.uniform(20, 40), 1)
            if stage == "4b":
                obs["sre_response"] = SCENARIO_META[14]["hitl_sre_response"]
        return obs


# ═══════════════════════════════════════════════════════════════
# PART C: PIPELINE STATE ENGINE
# ═══════════════════════════════════════════════════════════════

class PipelineStateEngine:
    """Tracks cascading state across pipeline stages."""

    def __init__(self):
        self._state = {}
        self.reset()

    def reset(self):
        self._state = {
            "pr_approved_with_risk": False,
            "risk_type": "",
            "build_failed": False,
            "build_was_flaky": False,
            "deploy_was_full_on_peak": False,
            "deployed_to_production": False,
            "hitl_triggered": False,
            "hitl_advice_type": "",
            "missed_migration": False,
        }

    def get(self) -> dict:
        return copy.deepcopy(self._state)

    def update_with_obs(self, stage, action: int, scenario_id: int, scenario_data: dict, obs: dict):
        """Mutate pipeline state based on agent actions, with observation context."""
        # After Stage 1
        if stage == 1:
            if action == 0:
                if scenario_data.get("is_security_risk"):
                    self._state["pr_approved_with_risk"] = True
                    self._state["risk_type"] = "security"
                if scenario_data.get("is_bug"):
                    self._state["pr_approved_with_risk"] = True
                    self._state["risk_type"] = "bug"
                if scenario_id == 5:
                    self._state["risk_type"] = "operational"
                if scenario_id == 11:
                    self._state["missed_migration"] = True

        # After Stage 2
        elif stage == 2:
            if action == 0 and obs.get("tests_failed", 0) > 0:
                self._state["build_failed"] = True
            if action == 0 and obs.get("build_status") == "flaky":
                self._state["build_was_flaky"] = True

        # After Stage 3
        elif stage == 3:
            if action == 0 and obs.get("traffic_level_pct", 0) > 85:
                self._state["deploy_was_full_on_peak"] = True
            if action == 0 and obs.get("deploy_environment") == "production":
                self._state["deployed_to_production"] = True

        # HITL trigger
        if action == 3 and stage in (3, 4):
            self._state["hitl_triggered"] = True
            self._state["hitl_advice_type"] = scenario_data.get("hitl_advice_type", "")

    def inject(self, obs_dict: dict, stage) -> dict:
        """Inject pipeline state effects into the observation."""
        if stage not in (4, "4b"):
            return obs_dict

        obs = copy.deepcopy(obs_dict)

        # Security risk propagation
        if self._state["pr_approved_with_risk"] and self._state["risk_type"] == "security":
            if "suspicious_activity" not in obs.get("active_alerts", []):
                obs.setdefault("active_alerts", [])
                if "suspicious_activity" not in obs["active_alerts"]:
                    obs["active_alerts"].append("suspicious_activity")

        # Peak deploy propagation
        if self._state["deploy_was_full_on_peak"]:
            obs["error_rate_pct"] = round(
                obs.get("error_rate_pct", 0) + random.uniform(2, 5), 2
            )
            obs["error_rate_pct"] = min(obs["error_rate_pct"], 100.0)
            obs["latency_p99_ms"] = round(
                obs.get("latency_p99_ms", 0) * random.uniform(1.5, 3.0), 1
            )
            obs.setdefault("active_alerts", [])
            if "high_traffic_error_spike" not in obs["active_alerts"]:
                obs["active_alerts"].append("high_traffic_error_spike")

        # Build failure propagation
        if self._state["build_failed"]:
            obs["error_rate_pct"] = round(
                obs.get("error_rate_pct", 0) + random.uniform(1, 3), 2
            )
            obs["error_rate_pct"] = min(obs["error_rate_pct"], 100.0)
            obs.setdefault("active_alerts", [])
            if "build_issue_in_production" not in obs["active_alerts"]:
                obs["active_alerts"].append("build_issue_in_production")

        # Missing migration propagation (from Scenario 11 stage 1 memory)
        if self._state.get("missed_migration"):
            obs.setdefault("active_alerts", [])
            if "P1: db_schema_mismatch_critical" not in obs["active_alerts"]:
                obs["active_alerts"].append("P1: db_schema_mismatch_critical")

        return obs


# ═══════════════════════════════════════════════════════════════
# PART D: GRADER ENGINE
# ═══════════════════════════════════════════════════════════════

class GraderEngine:
    """Deterministic per-stage reward computation."""

    def classify_correct(self, stage, action: int, obs: dict, scenario_data: dict) -> float:
        """Awards 0.10 for correct classification of the situation."""
        if stage == 1:
            if scenario_data.get("is_security_risk"):
                return 0.10 if action in (1, 3) else 0.0
            if scenario_data.get("is_bug"):
                return 0.10 if action in (1, 2, 3) else 0.0
            # Clean
            return 0.10 if action == 0 else 0.0

        elif stage == 2:
            tests_failed = obs.get("tests_failed", 0)
            build_status = obs.get("build_status", "pending")
            if tests_failed > 5:
                return 0.10 if action in (1, 3) else 0.0
            if build_status == "flaky" and tests_failed <= 4:
                return 0.10 if action == 2 else 0.0
            if build_status == "passing" and tests_failed == 0:
                return 0.10 if action == 0 else 0.0
            return 0.0

        elif stage == 3:
            traffic = obs.get("traffic_level_pct", 0)
            if traffic > 85:
                return 0.10 if action in (2, 3) else 0.0
            deploy_env = obs.get("deploy_environment", "staging")
            if deploy_env == "production":
                return 0.10 if action in (0, 2) else 0.0
            return 0.10 if action == 0 else 0.0

        elif stage in (4, "4b"):
            alerts = obs.get("active_alerts", [])
            has_p1 = any("P1" in str(a) for a in alerts)
            if has_p1:
                if action == 1: return 0.10  # Full partial for Rollback on P1
                if action == 3: return 0.05  # Slight credit for Escalating
                return 0.0
            if not alerts:
                if action == 0: return 0.10
                if action == 2: return 0.05
                return 0.0
            # Non-P1 alerts
            if action in (2, 3): return 0.10
            return 0.0

        return 0.0

    def risk_aware(self, stage, action: int, obs: dict, scenario_data: dict) -> float:
        """Awards 0.10 for correct risk identification."""
        is_security = scenario_data.get("is_security_risk", False)
        is_bug = scenario_data.get("is_bug", False)

        if is_security:
            if stage == 1:
                return 0.10 if action in (1, 3) else 0.0
            if stage in (4, "4b"):
                if action == 1: return 0.10
                if action == 3: return 0.05
                return 0.0
            # Other stages — partial risk awareness
            if action in (1, 2, 3): return 0.10
            return 0.0

        if is_bug:
            if stage in (1, 2):
                return 0.10 if action in (1, 2, 3) else 0.0
            # Later stages
            return 0.10 if action in (1, 2, 3) else 0.0

        # Clean scenario — not being over-cautious
        return 0.10 if action in (0, 2) else 0.0

    def optimal_action(self, stage, action: int, obs: dict,
                       scenario_data: dict, pipeline_state: dict) -> float:
        """Awards 0.15 ONLY for the single best action."""
        optimal_map = scenario_data.get("optimal", {})

        # Stage key normalisation (4b → "4b", integers for 1-4)
        stage_key = stage if stage != "4b" else "4b"

        # Special rules
        if stage == 3:
            traffic = obs.get("traffic_level_pct", 0)
            if traffic > 85 and not pipeline_state.get("build_failed"):
                return 0.15 if action == 2 else 0.0

        if stage in (4, "4b"):
            alerts = obs.get("active_alerts", [])
            has_p1 = any("P1" in str(a) for a in alerts)
            if has_p1:
                return 0.15 if action in (1, 3) else 0.0

        # Stage 4b HITL rules
        if stage == "4b":
            sre_response = obs.get("sre_response", "")
            hitl_advice = pipeline_state.get("hitl_advice_type", "")
            if hitl_advice == "rollback" or "rollback" in sre_response.lower():
                return 0.15 if action == 1 else 0.0
            if hitl_advice == "proceed":
                return 0.15 if action == 0 else 0.0
            return 0.15 if action == 0 else 0.0

        # Use the scenario optimal map
        optimal = optimal_map.get(stage_key)
        if optimal is not None:
            return 0.15 if action == optimal else 0.0

        # Default: approve is fine for clean stages
        return 0.15 if action == 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# PART E: DevOpsReleaseCmdEnv CLASS
# ═══════════════════════════════════════════════════════════════

class DevOpsReleaseCmdEnv:
    """Main OpenEnv environment class."""

    def __init__(self):
        self._scenario_engine = ScenarioEngine()
        self._pipeline_engine = PipelineStateEngine()
        self._grader = GraderEngine()
        self._current_obs: dict = {}
        self._current_stage = 1
        self._current_difficulty = 1
        self._current_scenario_id = 1
        self._hitl_active = False
        self._accumulated_reward = 0.0
        self._stages_run = 0
        self.reset()

    def reset(self, difficulty=None, seed=None) -> str:
        """
        Initialise a new episode.
        difficulty: 1=Easy, 2=Medium, 3=Hard, 4=Nightmare
        seed: for reproducibility
        Returns: JSON string of initial observation
        """
        if seed is not None:
            random.seed(seed)

        self._pipeline_engine.reset()
        self._accumulated_reward = 0.0
        self._stages_run = 0
        self._hitl_active = False

        if difficulty is None:
            difficulty = random.choice([1, 2, 3, 4])
        self._current_difficulty = difficulty

        # Assign scenario based on difficulty
        scenario_pools = {
            1: [1, 2],           # Easy: S01, S02
            2: [3, 4, 5, 12],    # Medium: S03, S04, S05, S12
            3: [6, 7, 8, 13],    # Hard: S06, S07, S08, S13
            4: [9, 10, 11, 14],  # Nightmare: S09, S10, S11, S14
        }
        self._current_scenario_id = random.choice(scenario_pools[difficulty])

        self._current_stage = 1
        self._generate_observation(stage=1)

        return json.dumps(self._current_obs)

    def _generate_observation(self, stage):
        """Generate and store observation for the given stage."""
        pipeline_state = self._pipeline_engine.get()
        obs_dict = self._scenario_engine.generate(
            self._current_scenario_id,
            stage,
            self._current_difficulty,
            pipeline_state,
        )
        # Inject pipeline state effects
        obs_dict = self._pipeline_engine.inject(obs_dict, stage)
        obs_dict["stage"] = stage
        obs_dict["difficulty"] = self._current_difficulty
        obs_dict["episode_id"] = (
            f"diff{self._current_difficulty}_s{self._current_scenario_id:02d}_seed"
        )
        self._current_obs = obs_dict

    def state(self) -> str:
        """Return current state without modification."""
        return json.dumps(self._current_obs)

    def step(self, action):
        """
        Process action and return (obs_json, reward, done, info).
        CRITICAL: Entire method body is wrapped in try/except.
        On ANY exception: return self.state(), 0.0, True, {}
        """
        try:
            info = {}  # ALWAYS empty — zero leakage

            # Validate action
            if not isinstance(action, int) or action not in (0, 1, 2, 3):
                return self.state(), 0.0, True, info

            stage = self._current_obs.get("stage", 1)

            # ── Catastrophic check FIRST ──────────────────────
            active_alerts = self._current_obs.get("active_alerts", [])
            if action == 0 and stage in (4, "4b"):
                if any("P1" in str(a) for a in active_alerts):
                    return self.state(), 0.0, True, info

            # ── Get scenario data for grading ─────────────────
            scenario_data = self._scenario_engine.get_scenario_data(
                self._current_scenario_id
            )
            pipeline_state = self._pipeline_engine.get()

            # ── Compute stage reward components ───────────────
            stage_reward = (
                self._grader.classify_correct(
                    stage, action, self._current_obs, scenario_data
                )
                + self._grader.risk_aware(
                    stage, action, self._current_obs, scenario_data
                )
                + self._grader.optimal_action(
                    stage, action, self._current_obs, scenario_data, pipeline_state
                )
            )
            self._accumulated_reward += stage_reward
            self._stages_run += 1

            # ── Update pipeline state ─────────────────────────
            self._pipeline_engine.update_with_obs(
                stage, action, self._current_scenario_id, scenario_data,
                self._current_obs,
            )

            # ── HITL trigger check ────────────────────────────
            hitl_scenarios = [7, 8, 9, 10, 11, 13, 14]
            if (
                action == 3
                and stage in (3, 4)
                and self._current_difficulty >= 3
                and self._current_scenario_id in hitl_scenarios
                and not self._hitl_active
            ):
                self._hitl_active = True
                self._current_stage = "4b"
                self._generate_observation("4b")
                partial = self._compute_final_reward(done=False)
                return self.state(), partial, False, info

            # ── Check if episode is done ──────────────────────
            max_stages = {1: 1, 2: 2, 3: 4, 4: 4}
            max_s = max_stages.get(self._current_difficulty, 1)

            # Episode ends if:
            #   action=1 (block/reject/rollback) — terminal
            #   all stages complete
            #   HITL stage just resolved
            if action == 1 or self._stages_run >= max_s or self._hitl_active:
                # Recovery bonus
                ps = self._pipeline_engine.get()
                if (ps.get("pr_approved_with_risk") or ps.get("build_failed") or ps.get("missed_migration")):
                    if action in (1, 3):
                        self._accumulated_reward += 0.15  # Buffed recovery bonus

                # Speed bonus
                if any("P1" in str(a) for a in active_alerts) and action == 1:
                    self._accumulated_reward += 0.05

                final_reward = self._compute_final_reward(done=True)
                return self.state(), final_reward, True, info

            # ── Advance to next stage ─────────────────────────
            if isinstance(stage, str):
                # "4b" → done
                final_reward = self._compute_final_reward(done=True)
                return self.state(), final_reward, True, info

            next_stage = int(stage) + 1
            if next_stage > 4:
                final_reward = self._compute_final_reward(done=True)
                return self.state(), final_reward, True, info

            self._current_stage = next_stage
            self._generate_observation(next_stage)

            partial = self._compute_final_reward(done=False)
            return self.state(), partial, False, info

        except Exception:
            return self.state(), 0.0, True, {}

    def _compute_final_reward(self, done: bool) -> float:
        """Normalise accumulated reward to [0.0, 1.0]."""
        if self._stages_run == 0:
            return 0.0
        raw_max = (0.35 * max(self._stages_run, 1)) + 0.15
        normalised = self._accumulated_reward / raw_max
        return round(max(0.0, min(1.0, normalised)), 2)
