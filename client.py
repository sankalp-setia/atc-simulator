from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from atc_advisor_env import ATCAction, ATCObservation, ATCState


class ATCAdvisorClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, difficulty: Optional[str] = None, seed: Optional[int] = None) -> ATCObservation:
        payload: Dict[str, Any] = {"difficulty": difficulty, "seed": seed}
        response = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()["observation"]
        return ATCObservation(**data)

    def step(self, action: ATCAction) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/step",
            json={
                "callsign": action.callsign,
                "command": action.command,
                "value": action.value,
                "runway": action.runway,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        data["observation"] = ATCObservation(**data["observation"])
        return data

    def state(self) -> ATCState:
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return ATCState(**response.json())

    def tasks(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def grader(self) -> Dict[str, float]:
        response = requests.get(f"{self.base_url}/grader", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def baseline(
        self,
        model: str = "Qwen/Qwen3-1.7B",
        seed: int = 7,
        provider: str = "heuristic",
        max_turns: int = 20,
    ) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/baseline",
            params={
                "model": model,
                "seed": seed,
                "provider": provider,
                "max_turns": max_turns,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
