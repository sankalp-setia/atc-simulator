from __future__ import annotations

import argparse
import json
import statistics
from typing import Any, Dict

import requests


def assert_score_range(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)):
        raise AssertionError(f"{name} must be numeric, got {type(value)}")
    if value < 0.0 or value > 1.0:
        raise AssertionError(f"{name} out of range [0.0, 1.0]: {value}")


def get_json(session: requests.Session, url: str, **kwargs: Any) -> Dict[str, Any]:
    resp = session.get(url, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp.json()


def post_json(session: requests.Session, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = session.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def validate(base_url: str, provider: str, model: str, seed: int, max_turns: int) -> Dict[str, Any]:
    session = requests.Session()
    out: Dict[str, Any] = {"base_url": base_url, "checks": {}}

    # 1) Health
    health = get_json(session, f"{base_url}/health")
    if health.get("status") != "healthy":
        raise AssertionError("/health did not return status=healthy")
    out["checks"]["health"] = "ok"

    # 2) Tasks + schema + minimum task count
    tasks_payload = get_json(session, f"{base_url}/tasks")
    tasks = tasks_payload.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) < 3:
        raise AssertionError("/tasks must return at least 3 tasks")
    if "action_schema" not in tasks_payload:
        raise AssertionError("/tasks missing action_schema")
    out["checks"]["tasks"] = {"count": len(tasks)}

    # 3) Reset
    reset_payload = post_json(session, f"{base_url}/reset", {"difficulty": "easy", "seed": seed})
    observation = reset_payload.get("observation", {})
    if "radar_text" not in observation:
        raise AssertionError("/reset observation missing radar_text")
    out["checks"]["reset"] = "ok"

    # 4) Step
    step_payload = post_json(
        session,
        f"{base_url}/step",
        {"callsign": "OWN", "command": "heading", "value": 180},
    )
    if "reward" not in step_payload or "done" not in step_payload:
        raise AssertionError("/step missing reward or done")

    info = step_payload.get("info", {})
    grader_scores = info.get("grader_scores", {})
    if not grader_scores:
        raise AssertionError("/step info missing grader_scores")
    for key in ["task_easy_safety", "task_medium_efficiency", "task_hard_phraseology", "overall"]:
        assert_score_range(f"grader_scores.{key}", grader_scores.get(key))
    out["checks"]["step"] = "ok"

    # 5) State
    state_payload = get_json(session, f"{base_url}/state")
    if "gym_obs" not in state_payload or "scenario_difficulty" not in state_payload:
        raise AssertionError("/state missing required fields")
    out["checks"]["state"] = "ok"

    # 6) Grader endpoint
    grader_payload = get_json(session, f"{base_url}/grader")
    for key in ["task_easy_safety", "task_medium_efficiency", "task_hard_phraseology", "overall"]:
        assert_score_range(f"/grader.{key}", grader_payload.get(key))
    out["checks"]["grader"] = "ok"

    # 7) Baseline endpoint (defaults safe to heuristic for offline reproducibility)
    baseline_payload = get_json(
        session,
        f"{base_url}/baseline",
        params={"provider": provider, "model": model, "seed": seed, "max_turns": max_turns},
    )

    assert_score_range("baseline.overall_score", baseline_payload.get("overall_score"))
    baseline_tasks = baseline_payload.get("tasks", {})
    if not isinstance(baseline_tasks, dict) or len(baseline_tasks) < 3:
        raise AssertionError("/baseline tasks must include at least 3 scored tasks")
    for task_id, score_obj in baseline_tasks.items():
        assert_score_range(f"baseline.tasks.{task_id}.task_score", score_obj.get("task_score"))
        assert_score_range(f"baseline.tasks.{task_id}.easy", score_obj.get("easy"))
        assert_score_range(f"baseline.tasks.{task_id}.medium", score_obj.get("medium"))
        assert_score_range(f"baseline.tasks.{task_id}.hard", score_obj.get("hard"))
    out["checks"]["baseline"] = "ok"

    out["status"] = "passed"
    return out


def validate_multi_seed(base_url: str, provider: str, model: str, seeds: list[int], max_turns: int) -> Dict[str, Any]:
    session = requests.Session()
    runs: list[Dict[str, Any]] = []
    overall_scores: list[float] = []
    task_scores: Dict[str, list[float]] = {}

    for seed in seeds:
        payload = get_json(
            session,
            f"{base_url}/baseline",
            params={"provider": provider, "model": model, "seed": seed, "max_turns": max_turns},
        )

        overall = float(payload.get("overall_score", 0.0))
        assert_score_range("baseline.overall_score", overall)
        overall_scores.append(overall)

        tasks = payload.get("tasks", {})
        if not isinstance(tasks, dict):
            raise AssertionError("/baseline tasks must be an object")

        run_row: Dict[str, Any] = {"seed": seed, "overall": overall}
        for task_id, score_obj in tasks.items():
            task_score = float(score_obj.get("task_score", 0.0))
            assert_score_range(f"baseline.tasks.{task_id}.task_score", task_score)
            task_scores.setdefault(task_id, []).append(task_score)
            run_row[task_id] = task_score
        runs.append(run_row)

    summary: Dict[str, Any] = {
        "seeds": seeds,
        "overall_mean": statistics.mean(overall_scores) if overall_scores else 0.0,
        "overall_std": statistics.pstdev(overall_scores) if len(overall_scores) > 1 else 0.0,
        "tasks": {},
        "runs": runs,
    }

    for task_id, values in task_scores.items():
        summary["tasks"][task_id] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-submission validator for atc-advisor-v0")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--provider", type=str, default="heuristic")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default="7,11,19")
    parser.add_argument("--max-turns", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    result = validate(
        base_url=args.base_url.rstrip("/"),
        provider=args.provider,
        model=args.model,
        seed=args.seed,
        max_turns=args.max_turns,
    )
    result["multi_seed_baseline"] = validate_multi_seed(
        base_url=args.base_url.rstrip("/"),
        provider=args.provider,
        model=args.model,
        seeds=parsed_seeds,
        max_turns=args.max_turns,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
