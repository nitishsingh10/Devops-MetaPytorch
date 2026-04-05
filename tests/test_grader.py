"""
DevOps Release Commander — Grader Variance & Safety Tests
==========================================================
Run: python3 tests/test_grader.py
All 6 tests must pass before submission.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment import DevOpsReleaseCmdEnv


def test_grader_variance():
    """
    CRITICAL: This test MUST pass before any submission.
    Checks that graders return different rewards for different actions.
    If any scenario returns the same reward for all 4 actions = DISQUALIFICATION.
    """
    env = DevOpsReleaseCmdEnv()
    all_passed = True

    scenario_difficulty_map = {
        1: 1, 2: 1,
        3: 2, 4: 2, 5: 2,
        6: 3, 7: 3, 8: 3,
        9: 4, 10: 4, 11: 4,
    }

    print("Running grader variance test...")

    for scenario_id in range(1, 12):
        diff = scenario_difficulty_map[scenario_id]
        rewards = []

        for action in [0, 1, 2, 3]:
            env.reset(difficulty=diff, seed=42)
            # Override the randomly selected scenario with the specific test scenario
            env._current_scenario_id = scenario_id
            env._current_stage = 1
            env._generate_observation(stage=1)

            _, reward, _, _ = env.step(action)
            rewards.append(reward)

        unique_rewards = len(set(rewards))
        status = "PASS" if unique_rewards >= 2 else "FAIL"
        if unique_rewards < 2:
            all_passed = False
        print(
            f"  Scenario S{scenario_id:02d} (diff={diff}): "
            f"rewards={rewards} unique={unique_rewards} → {status}"
        )

    print()
    if all_passed:
        print("GRADER VARIANCE TEST: PASSED ✅")
        print("All 11 scenarios show reward variance — safe to submit.")
    else:
        print("GRADER VARIANCE TEST: FAILED ❌")
        print("STOP — do not submit. Fix grader variance before proceeding.")
        sys.exit(1)


def test_no_negative_rewards():
    """Reward must always be in [0.0, 1.0]."""
    env = DevOpsReleaseCmdEnv()
    print("Running reward bounds test...")
    for diff in [1, 2, 3, 4]:
        for action in [-1, 0, 1, 2, 3, 99, None, "bad"]:
            env.reset(difficulty=diff, seed=42)
            _, reward, _, info = env.step(action)
            assert 0.0 <= reward <= 1.0, (
                f"FAIL: diff={diff} action={action} reward={reward}"
            )
            assert info == {}, f"FAIL: info not empty: {info}"
    print("Reward bounds test: PASSED ✅")


def test_step_never_crashes():
    """step() must never raise an exception for any input."""
    env = DevOpsReleaseCmdEnv()
    print("Running crash safety test...")
    bad_inputs = [None, "string", [], {}, -999, 999, 3.14, True, False]
    for bad in bad_inputs:
        env.reset(difficulty=1, seed=42)
        try:
            result = env.step(bad)
            assert len(result) == 4, "Must return 4-tuple"
        except Exception as e:
            print(f"FAIL: step({bad!r}) raised {e}")
            sys.exit(1)
    print("Crash safety test: PASSED ✅")


def test_determinism():
    """seed=42 must produce identical observations on repeated calls."""
    env = DevOpsReleaseCmdEnv()
    print("Running determinism test...")
    for diff in [1, 2, 3, 4]:
        obs_list = [env.reset(difficulty=diff, seed=42) for _ in range(3)]
        assert len(set(obs_list)) == 1, f"FAIL: diff={diff} not deterministic"
    print("Determinism test: PASSED ✅")


def test_state_matches_reset():
    """state() must return same obs as reset() without modification."""
    env = DevOpsReleaseCmdEnv()
    print("Running state() consistency test...")
    for diff in [1, 2, 3, 4]:
        obs = env.reset(difficulty=diff, seed=42)
        assert obs == env.state(), f"FAIL: state() != reset() for diff={diff}"
    print("State consistency test: PASSED ✅")


def test_hitl_triggered_on_action_3():
    """action=3 at Stage 4 in Hard+ must trigger HITL (done=False)."""
    import random

    env = DevOpsReleaseCmdEnv()
    print("Running HITL trigger test...")
    env.reset(difficulty=3, seed=42)
    
    # Fast-forward to Stage 4 Memory Leak scenario
    env._current_scenario_id = 7  # S07 — Memory Leak
    env._current_stage = 4
    env._generate_observation(stage=4)
    # Force alerts to not be P1 so we don't hit catastrophic
    env._current_obs["active_alerts"] = ["high_memory_usage"]
    _, _, done, _ = env.step(3)  # action=3 = Escalate
    assert done is False, "FAIL: HITL should set done=False"
    assert (
        env._current_obs.get("sre_response") is not None
    ), "FAIL: sre_response missing"
    print("HITL trigger test: PASSED ✅")


if __name__ == "__main__":
    test_grader_variance()
    test_no_negative_rewards()
    test_step_never_crashes()
    test_determinism()
    test_state_matches_reset()
    test_hitl_triggered_on_action_3()
    print()
    print("ALL TESTS PASSED — safe to proceed to submission.")
