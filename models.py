from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
	from openenv import Action, Observation, State
except ImportError:
	class Action:  # pragma: no cover
		pass

	class Observation:  # pragma: no cover
		pass

	class State:  # pragma: no cover
		pass


class ATCAction(BaseModel, Action):
	"""Typed action schema for one ATC clearance."""

	model_config = ConfigDict(extra="forbid")

	callsign: str = Field(default="OWN", min_length=1)
	command: str = Field(default="hold", min_length=1)
	value: Optional[float] = None
	runway: Optional[str] = None


class ATCObservation(BaseModel, Observation):
	"""Typed observation schema returned by reset/step."""

	model_config = ConfigDict(extra="forbid")

	radar_text: str
	messages: List[Dict[str, str]] = Field(default_factory=list)
	conflicts: int = 0
	active_aircraft: int = 0


class ATCState(BaseModel, State):
	"""Typed internal state schema for debug/grading use."""

	model_config = ConfigDict(extra="forbid")

	gym_obs: Dict[str, Any] = Field(default_factory=dict)
	scenario_difficulty: str = "medium"


class ATCReward(BaseModel):
	"""Typed reward decomposition model (0-1 normalized components)."""

	model_config = ConfigDict(extra="forbid")

	total: float
	safety: float
	efficiency: float
	compliance: float
	track_alignment: float
	progress_score: float
	goal: float
	gym: float


__all__ = ["ATCAction", "ATCObservation", "ATCState", "ATCReward"]
