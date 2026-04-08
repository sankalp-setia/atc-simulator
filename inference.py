from __future__ import annotations

import argparse
import os
import re
import time
from typing import Any, Dict, List

from openai import OpenAI

from atc_advisor_env import ATCAdvisorEnv, extract_clearance
from baseline_inference import DEFAULT_SYSTEM_PROMPT, TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_API_RETRIES = int(os.getenv("MAX_API_RETRIES", "3"))
DEFAULT_RETRY_SECONDS = float(os.getenv("DEFAULT_RETRY_SECONDS", "5"))

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _fmt_reward(value: float) -> str:
    return f"{float(value):.2f}"


def _format_action_for_log(action: Dict[str, Any]) -> str:
    callsign = str(action.get("callsign", "OWN"))
    command = str(action.get("command", "hold"))
    value = action.get("value")
    runway = action.get("runway")

    parts: List[str] = [callsign, command]
    if value is not None:
        value_f = float(value)
        if value_f.is_integer():
            parts.append(str(int(value_f)))
        else:
            parts.append(f"{value_f:.1f}")
    if runway is not None:
        parts.extend(["runway", str(runway)])
    return "[clear " + " ".join(parts) + "]"


def _build_user_prompt(observation_text: str, turn: int) -> str:
    return (
        f"Turn {turn}.\n"
        "Issue one ATC clearance in bracket format.\n"
        f"Radar:\n{observation_text}\n"
    )


def _task_difficulty(task_name: str) -> str:
    for task in TASKS:
        if task.get("id") == task_name:
            return str(task.get("difficulty", "medium"))
    return "medium"


def _ordered_tasks() -> List[Dict[str, Any]]:
    rank = {"easy": 0, "medium": 1, "hard": 2}
    return sorted(TASKS, key=lambda t: rank.get(str(t.get("difficulty", "medium")), 99))


def _resolve_tasks(task_name: str) -> List[Dict[str, Any]]:
    if task_name in {"all", "*"}:
        return _ordered_tasks()

    for task in TASKS:
        if str(task.get("id")) == task_name:
            return [task]

    valid = ", ".join(str(task.get("id")) for task in _ordered_tasks())
    raise ValueError(f"Unknown task '{task_name}'. Valid values: all, {valid}")


def _strict_score(value: float) -> float:
    v = float(max(0.0, min(1.0, float(value))))
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    return float(v)


def _retry_after_seconds(error_text: str) -> float:
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return DEFAULT_RETRY_SECONDS


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _generate_with_retry(client: OpenAI, prompt: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit_error(exc) or attempt >= MAX_API_RETRIES:
                raise
            time.sleep(_retry_after_seconds(str(exc)))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to generate completion")


def run_episode(task_name: str, benchmark: str, max_steps: int, seed: int) -> int:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    difficulty = _task_difficulty(task_name)
    env = ATCAdvisorEnv(default_difficulty=difficulty, seed=seed)

    rewards: List[float] = []
    step_count = 0
    done = False
    success = False
    fatal_error = False
    final_score = 0.01

    print(f"[START] task={task_name} difficulty={difficulty} env={benchmark} model={MODEL_NAME}")

    try:
        obs = env.reset(seed=seed)

        while not done and step_count < max_steps:
            step_count += 1
            last_action_error: str | None = None
            reward = 0.0
            action_str = "[clear OWN hold 0]"

            try:
                prompt = _build_user_prompt(obs.radar_text, step_count)
                completion_text = _generate_with_retry(client, prompt)
                action = extract_clearance(completion_text)
                action_payload = action.model_dump(exclude_none=True)
                action_str = _format_action_for_log(action_payload)

                obs, reward, done, info = env.step(action)
                info_dict = info.get("info", {}) if isinstance(info, dict) else {}
                raw_error = info_dict.get("last_action_error")
                if raw_error is not None:
                    last_action_error = str(raw_error)
                grader_scores = info.get("grader_scores", {}) if isinstance(info, dict) else {}
                final_score = _strict_score(float(grader_scores.get("overall", final_score)))
            except Exception as exc:
                fatal_error = True
                done = True
                _ = str(exc)

            rewards.append(float(reward))
            error_text = last_action_error if last_action_error is not None else "null"
            print(
                f"[STEP] step={step_count} action={action_str} reward={_fmt_reward(reward)} "
                f"done={_bool_text(done)} error={error_text}"
            )

        success = done and not fatal_error
    except Exception:
        success = False
    finally:
        env.close()
        rewards_text = ",".join(_fmt_reward(r) for r in rewards)
        print(
            f"[END] success={_bool_text(success)} steps={step_count} score={_fmt_reward(final_score)} rewards={rewards_text}"
        )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hackathon inference runner")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--env", type=str, default="atc-advisor-v0")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks_to_run = _resolve_tasks(args.task)
    for i, task in enumerate(tasks_to_run):
        run_episode(
            task_name=str(task["id"]),
            benchmark=args.env,
            max_steps=args.max_steps,
            seed=args.seed + i,
        )
    raise SystemExit(0)
