# atc-advisor-v0

`atc-advisor-v0` is a real-world OpenEnv environment for **air traffic merge advisory**. The agent acts as an AI co-pilot to issue clearances for merging arrivals on approach while balancing safety, efficiency, and phraseology compliance.

## Why this is real-world

Air traffic merge management is a real controller workflow:
- maintain separation (conflict avoidance),
- sequence arrivals efficiently,
- communicate compliant, structured clearances.

This environment models that workflow using BlueSky-Gym `MergeEnv-v0` under OpenEnv-style APIs.

## API surface

- `reset(seed=None) -> ATCObservation`
- `step(action) -> (ATCObservation, reward, done, info)`
- `state() -> ATCState`
- `close()`

Main implementation: `atc_advisor_env.py`

OpenEnv-style structure aliases are also provided:
- `models.py`
- `client.py`
- `server/environment.py`
- `server/app.py`

## Typed models

- `ATCAction`
  - `callsign: str`
  - `command: str` (`heading|speed|altitude|takeoff|land|hold|vector`)
  - `value: Optional[float]`
  - `runway: Optional[str]`
- `ATCObservation`
  - `radar_text: str`
  - `messages: List[Dict[str, str]]`
  - `conflicts: int`
  - `active_aircraft: int`
- `ATCState`
  - `gym_obs: Dict[str, Any]`
  - `scenario_difficulty: str`
- `ATCReward`
  - `total, safety, efficiency, compliance, track_alignment, progress_score, goal, gym`

All models are Pydantic schemas in `models.py`.

## Tasks and graders (0.0 to 1.0)

The environment exposes deterministic grader components in `info["grader_scores"]`:

1. `task_easy_safety` (easy)
- Objective: avoid conflicts with light traffic.
- Score: conflict-based safety score in `[0, 1]`.

2. `task_medium_efficiency` (medium)
- Objective: progress efficiently while preserving safety.
- Score: combined efficiency + goal progression in `[0, 1]`.

3. `task_hard_phraseology` (hard)
- Objective: maintain compliant command structure under pressure.
- Score: combined compliance + safety in `[0, 1]`.

`overall` is the mean task score with terminal conflict penalty.

## Reward shaping

Trajectory reward combines:
- safety,
- efficiency,
- compliance,
- goal progress.

This gives dense feedback across steps and discourages poor behavior through conflict penalties and compliance checks.

Paper-informed additions for `MergeEnv-v0`:
- explicit progress proxy toward FAF,
- track-alignment term (heading vs FAF/waypoint bearing),
- KPI export in step `info["kpis"]` for:
  - average track error,
  - intrusions/conflicts,
  - steps-to-FAF proxy,
  - remaining FAF distance proxy.

These align with the paper's recommendation to evaluate beyond reward (e.g., track deviation, intrusions, time-to-waypoint).

## Training modules

- `training/rollout.py`:
  - `rollout_func` compatible with GRPO-style loops.
- `training/rewards.py`:
  - `reward_safety`,
  - `reward_efficiency`,
  - `reward_compliance`,
  - `reward_final_goal`.

Minimal GRPO runner:
- `training/run_grpo.py`

Example launch:

```bash
python training/run_grpo.py \
  --model Qwen/Qwen3-1.7B \
  --difficulty medium \
  --dataset-size 128 \
  --epochs 1 \
  --output-dir outputs/atc-grpo
```

## Baseline inference

`baseline_inference.py` uses OpenAI API and reads credentials from `OPENAI_API_KEY`.

It also supports local non-OpenAI providers:
- `--provider vllm` (recommended for local GPU)
- `--provider transformers` (HF model inference)
- `--provider heuristic` (no external model, smoke test)

Run baseline over all 3 tasks:

```bash
python baseline_inference.py --provider heuristic
python baseline_inference.py --provider vllm --model Qwen/Qwen3-1.7B
python baseline_inference.py --provider transformers --model Qwen/Qwen3-1.7B
```

Determinism settings:
- fixed `seed` (default `7`),
- deterministic decoding (`temperature=0` or `do_sample=False`).

## Hackathon inference entrypoint

Submission includes root-level `inference.py` for evaluator execution.

Required environment variables:
- `API_BASE_URL` (default: `https://api.openai.com/v1`)
- `MODEL_NAME` (default: `gpt-4.1-mini`)
- `HF_TOKEN` (required)

Run:

```bash
HF_TOKEN=<token> python inference.py --task task_easy_safety --env atc-advisor-v0 --max-steps 10 --seed 7
```

Output format follows the required line contract:
- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

## Validator endpoints

FastAPI server in `server/app.py` exposes:
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader`
- `GET /baseline`

## Pre-submission validator

Run against a locally running server:

```bash
python scripts/pre_submission_validate.py --base-url http://localhost:8000 --provider heuristic
```

This checks:
- health endpoint
- task enumeration (>= 3 tasks)
- reset/step/state lifecycle
- grader score range enforcement (0.0 to 1.0)
- baseline endpoint reproducibility output shape and score ranges

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t atc-advisor-v0 .
docker run --rm -p 8000:8000 atc-advisor-v0
```

## Hugging Face Spaces

1. Push this repository to a Docker Space.
2. Keep `openenv.yaml` at repo root.
3. Ensure Space has environment variable `OPENAI_API_KEY` if calling `/baseline`.
4. Add `openenv` tag in Space metadata.

## Example baseline score table format

Use this format in your submission report:

| Task | Score |
|------|-------|
| task_easy_safety | 0.xx |
| task_medium_efficiency | 0.xx |
| task_hard_phraseology | 0.xx |
| overall | 0.xx |

## OpenEnv manifest

`openenv.yaml` is included and points to `atc_advisor_env:ATCAdvisorEnv`.


ATC GRPO Training Lifecycle (End-to-End)
1. Components
Trainer orchestration: run_grpo.py
Rollout adapter: rollout.py
Environment + reward + parser: atc_advisor_env.py
Reward hooks: rewards.py
2. Single Training Iteration Flow
GRPOTrainer selects a prompt from dataset.
Rollout function starts one episode in ATCAdvisorEnv.
Environment reset creates MergeEnv-v0 scenario and returns radar text observation.
Prompt builder adds radar snapshot plus conversation history.
Model generates one clearance text.
Parser converts clearance text into ATCAction.
Action mapper converts ATCAction into gym action format (Box or Discrete fallback).
MergeEnv-v0 executes one simulator step.
Wrapper computes shaped reward parts:
safety
efficiency
compliance
track alignment
goal
Wrapper computes task graders and KPIs.
Rollout stores token ids, logprobs, and reward channels.
Loop repeats until done or max turns.
GRPO uses reward channels to compute policy update.
Model parameters are updated.
3. Inference/Evaluation Flow
Baseline or API caller gets current observation.
Model or heuristic emits clearance text.
Parser maps text to ATCAction.
Environment steps simulator.
Response returns:
next radar snapshot
shaped reward
done flag
grader scores
KPI metrics
4. Action Translation Rules
heading and vector commands map to normalized heading delta.
speed command maps to normalized speed delta.
hold applies mild slow-down proxy.
land, takeoff, altitude are currently phraseology-valid but minimally actuated in this adapter.
5. Reward Design Summary
Total reward is weighted sum of safety, efficiency, compliance, track alignment, and goal.
This creates dense training feedback and avoids sparse binary-only success signals.

6. Why This Works For ATC
Simulator gives realistic traffic dynamics.
Wrapper gives LLM-friendly text interface.
Reward shaping aligns learning with operational priorities.
Graders provide hackathon-compatible, bounded scoring outputs.
