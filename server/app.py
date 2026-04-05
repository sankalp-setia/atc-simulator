from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from atc_advisor_env import ATCAction, ATCAdvisorEnv
from baseline_inference import TASKS, run_baseline


class StepActionModel(BaseModel):
    callsign: str = Field(default="OWN")
    command: str = Field(default="hold")
    value: Optional[float] = Field(default=None)
    runway: Optional[str] = Field(default=None)


class ResetRequest(BaseModel):
    difficulty: Optional[str] = Field(default=None)
    seed: Optional[int] = Field(default=None)


class MessageModel(BaseModel):
    category: str
    content: str


class ObservationModel(BaseModel):
    radar_text: str
    messages: List[MessageModel]
    conflicts: int
    active_aircraft: int


class HealthResponseModel(BaseModel):
    status: str


class TaskModel(BaseModel):
    id: str
    difficulty: str
    objective: str


class TasksResponseModel(BaseModel):
    tasks: List[TaskModel]
    action_schema: Dict[str, Any]


class ResetResponseModel(BaseModel):
    observation: ObservationModel


class StepResponseModel(BaseModel):
    observation: ObservationModel
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponseModel(BaseModel):
    gym_obs: Dict[str, Any]
    scenario_difficulty: str


class GraderResponseModel(BaseModel):
    task_easy_safety: float
    task_medium_efficiency: float
    task_hard_phraseology: float
    overall: float


class BaselineTaskScoreModel(BaseModel):
    task_score: float
    easy: float
    medium: float
    hard: float
    turns: float


class BaselineResponseModel(BaseModel):
    provider: str
    model: str
    seed: int
    max_turns: int
    overall_score: float
    tasks: Dict[str, BaselineTaskScoreModel]


app = FastAPI(title="ATC Advisor OpenEnv API", version="0.1.0")
_env = ATCAdvisorEnv()
_last_grader: Dict[str, float] = {
    "task_easy_safety": 0.0,
    "task_medium_efficiency": 0.0,
    "task_hard_phraseology": 0.0,
    "overall": 0.0,
}


@app.get("/health", response_model=HealthResponseModel)
def health() -> HealthResponseModel:
    return {"status": "healthy"}


@app.get("/tasks", response_model=TasksResponseModel)
def tasks() -> TasksResponseModel:
    return {
        "tasks": TASKS,
        "action_schema": {
            "type": "object",
            "required": ["callsign", "command"],
            "properties": {
                "callsign": {"type": "string", "example": "AA123"},
                "command": {
                    "type": "string",
                    "enum": ["heading", "speed", "altitude", "takeoff", "land", "hold", "vector"],
                },
                "value": {"type": ["number", "null"], "example": 180},
                "runway": {"type": ["string", "null"], "example": "27L"},
            },
        },
    }


@app.post("/reset", response_model=ResetResponseModel)
def reset(req: ResetRequest = ResetRequest()) -> ResetResponseModel:
    global _env
    if req.difficulty:
        _env.close()
        _env = ATCAdvisorEnv(default_difficulty=req.difficulty, seed=req.seed)
    obs = _env.reset(seed=req.seed)
    return {
        "observation": {
            "radar_text": obs.radar_text,
            "messages": obs.messages,
            "conflicts": obs.conflicts,
            "active_aircraft": obs.active_aircraft,
        }
    }


@app.post("/step", response_model=StepResponseModel)
def step(action: StepActionModel) -> StepResponseModel:
    global _last_grader
    obs, reward, done, info = _env.step(
        ATCAction(
            callsign=action.callsign,
            command=action.command,
            value=action.value,
            runway=action.runway,
        )
    )
    _last_grader = info.get("grader_scores", _last_grader)
    return {
        "observation": {
            "radar_text": obs.radar_text,
            "messages": obs.messages,
            "conflicts": obs.conflicts,
            "active_aircraft": obs.active_aircraft,
        },
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=StateResponseModel)
def state() -> StateResponseModel:
    st = _env.state()
    return {
        "gym_obs": st.gym_obs,
        "scenario_difficulty": st.scenario_difficulty,
    }


@app.get("/grader", response_model=GraderResponseModel)
def grader() -> GraderResponseModel:
    return _last_grader


@app.get("/baseline")
def baseline(
    model: str = "Qwen/Qwen3-1.7B",
    seed: int = 7,
    provider: str = "heuristic",
    max_turns: int = 20,
) -> BaselineResponseModel:
    return run_baseline(model=model, seed=seed, provider=provider, max_turns=max_turns)


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
