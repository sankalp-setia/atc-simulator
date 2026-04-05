from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from models import ATCAction, ATCObservation, ATCReward, ATCState

try:
    import bluesky_gym
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "bluesky-gym is required. Install with: pip install bluesky-gym"
    ) from exc

bluesky_gym.register_envs()


class ATCAdvisorEnv:
    """
    OpenEnv-compatible ATC environment wrapper for BlueSky-Gym MergeEnv-v0.

    API:
      - reset(seed=None) -> ATCObservation
      - step(action) -> (ATCObservation, reward, done, info)
      - state() -> ATCState
      - close()
    """

    CLEARANCE_PATTERN = re.compile(
        r"\[\s*clear\s+([A-Za-z0-9\-]+)\s+([a-zA-Z]+)(?:\s+([\-]?[0-9]+(?:\.[0-9]+)?))?(?:\s+runway\s+([A-Za-z0-9]+))?.*\]",
        re.IGNORECASE,
    )

    def __init__(self, default_difficulty: Optional[str] = None, seed: Optional[int] = None):
        self.gym_env: Optional[gym.Env] = None
        self.default_difficulty = default_difficulty
        self.rng = random.Random(seed)

        self.difficulty_levels: Dict[str, Dict[str, Union[int, float]]] = {
            "easy": {"n_ac": 4, "max_steps": 100, "goal_bonus": 0.30},
            "medium": {"n_ac": 7, "max_steps": 150, "goal_bonus": 0.40},
            "hard": {"n_ac": 12, "max_steps": 200, "goal_bonus": 0.50},
        }

        self.current_scenario: Dict[str, Any] = {
            "difficulty": "medium",
            "step": 0,
            "max_steps": 150,
        }
        self.last_obs: Dict[str, Any] = {}
        self.last_info: Dict[str, Any] = {}
        self.cumulative_reward: float = 0.0
        self.history: List[Dict[str, str]] = []
        self.prev_distance_to_faf: Optional[float] = None
        self.kpis: Dict[str, float] = {
            "avg_track_error_deg": 0.0,
            "intrusions": 0.0,
            "steps_to_faf_proxy": 0.0,
            "distance_to_faf": 0.0,
        }
        self._track_error_accum = 0.0
        self._track_error_count = 0

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> ATCObservation:
        difficulty = self.default_difficulty or self.rng.choice(["easy", "medium", "hard"])
        if difficulty not in self.difficulty_levels:
            difficulty = "medium"

        params = self.difficulty_levels[difficulty]
        n_ac = int(params["n_ac"])

        if self.gym_env is not None:
            self.gym_env.close()

        try:
            self.gym_env = gym.make("MergeEnv-v0", render_mode=None, n_ac=n_ac) #Constructor returns wrapped env object with observation_space and action_space.
        except TypeError:
            # Some BlueSky-Gym versions may not expose n_ac override in gym.make.
            self.gym_env = gym.make("MergeEnv-v0", render_mode=None)

        gym_obs, info = self.gym_env.reset(seed=seed)
        self.last_obs = self._obs_to_dict(gym_obs)
        self.last_info = info or {}
        self.cumulative_reward = 0.0
        self.prev_distance_to_faf = self._extract_distance_to_faf(self.last_obs)
        self._track_error_accum = 0.0
        self._track_error_count = 0
        self.kpis = {
            "avg_track_error_deg": 0.0,
            "intrusions": 0.0,
            "steps_to_faf_proxy": 0.0,
            "distance_to_faf": float(self.prev_distance_to_faf or 0.0),
        }

        self.current_scenario = {
            "difficulty": difficulty,
            "step": 0,
            "max_steps": int(params["max_steps"]),
        }

        self.history = [
            {
                "category": "SYSTEM",
                "content": (
                    f"New scenario ({difficulty}) started. Task: merge arrivals safely to final approach "
                    "while maintaining efficient sequencing and phraseology compliance."
                ),
            }
        ]

        radar_text = self._obs_to_radar_text(self.last_obs)
        return ATCObservation(
            radar_text=radar_text,
            conflicts=0,
            active_aircraft=self._active_aircraft_count(self.last_obs, self.last_info),
            messages=list(self.history),
        )

    def step(self, action: Union[ATCAction, str]) -> Tuple[ATCObservation, float, bool, Dict[str, Any]]:
        if self.gym_env is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        parsed = self._coerce_action(action)
        gym_action = self._parse_clearance_to_gym_action(parsed)

        step_result = self.gym_env.step(gym_action)
        if len(step_result) == 5:
            gym_obs, gym_reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            gym_obs, gym_reward, done, info = step_result
            truncated = False

        self.last_obs = self._obs_to_dict(gym_obs)
        self.last_info = info or {}

        self.current_scenario["step"] += 1
        done = bool(done or self.current_scenario["step"] >= self.current_scenario["max_steps"])

        conflicts = int(self.last_info.get("conflicts", 0)) # Gym env should provide this in info; fallback to 0 if missing.
        dist_to_faf = self._extract_distance_to_faf(self.last_obs)
        track_error = self._extract_track_error_deg(self.last_obs)
        progress_score = self._compute_progress_score(dist_to_faf)

        reward, reward_parts = self._compute_grader_reward(
            gym_reward=float(gym_reward),
            conflicts=conflicts,
            done=done,
            action=parsed,
            progress_score=progress_score,
            track_error_deg=track_error,
        )
        self.cumulative_reward += reward

        self._track_error_accum += track_error
        self._track_error_count += 1
        self.kpis = {
            "avg_track_error_deg": self._track_error_accum / max(self._track_error_count, 1),
            "intrusions": float(conflicts),
            "steps_to_faf_proxy": float(self.current_scenario["step"]),
            "distance_to_faf": float(dist_to_faf if dist_to_faf is not None else 0.0),
        }

        grader_scores = self._agent_graders(done=done, conflicts=conflicts, reward_parts=reward_parts)

        action_text = self._format_action(parsed)
        outcome_text = (
            f"step={self.current_scenario['step']} conflicts={conflicts} "
            f"reward={reward:.3f} (safety={reward_parts['safety']:.2f}, "
            f"efficiency={reward_parts['efficiency']:.2f}, compliance={reward_parts['compliance']:.2f}, "
            f"goal={reward_parts['goal']:.2f})"
        )

        self.history.append({"category": "CLEARANCE", "content": action_text})
        self.history.append({"category": "OUTCOME", "content": outcome_text})
        self.history = self.history[-12:]

        obs = ATCObservation(
            radar_text=self._obs_to_radar_text(self.last_obs),
            conflicts=conflicts,
            active_aircraft=self._active_aircraft_count(self.last_obs, self.last_info),
            messages=list(self.history),
        )

        reward_model = ATCReward(
            total=float(reward),
            safety=float(reward_parts["safety"]),
            efficiency=float(reward_parts["efficiency"]),
            compliance=float(reward_parts["compliance"]),
            track_alignment=float(reward_parts["track_alignment"]),
            progress_score=float(reward_parts["progress_score"]),
            goal=float(reward_parts["goal"]),
            gym=float(reward_parts["gym"]),
        )

        info_out: Dict[str, Any] = {
            "info": self.last_info,
            "truncated": truncated,
            "difficulty": self.current_scenario["difficulty"],
            "reward_parts": reward_parts,
            "reward_model": reward_model.model_dump(),
            "grader_scores": grader_scores,
            "kpis": self.kpis,
            "episode_step": self.current_scenario["step"],
            "cumulative_reward": self.cumulative_reward,
        }

        return obs, reward, done, info_out

    def state(self) -> ATCState:
        gym_obs = self._obs_to_dict(self.last_obs)
        return ATCState(
            gym_obs=gym_obs,
            scenario_difficulty=self.current_scenario.get("difficulty", "medium"),
        )

    def close(self) -> None:
        if self.gym_env is not None:
            self.gym_env.close()
            self.gym_env = None

    def _obs_to_radar_text(self, obs: Dict[str, Any]) -> str:
        own = self._extract_ownship(obs)
        traffic = self._extract_traffic(obs)

        lines = ["RADAR SNAPSHOT:"]
        lines.append(
            "Ownship {callsign}: hdg {hdg:.0f} deg, spd {spd:.0f} kts, alt {alt:.0f} ft".format(
                callsign=own.get("callsign", "OWN"),
                hdg=float(own.get("heading", 0.0)),
                spd=float(own.get("speed", 0.0)),
                alt=float(own.get("altitude", 0.0)),
            )
        )

        if not traffic:
            lines.append("Traffic: none reported")
        else:
            lines.append(f"Traffic ({len(traffic)}):")
            for ac in traffic[:8]:
                lines.append(
                    "- {callsign}: hdg {hdg:.0f} deg, spd {spd:.0f} kts, alt {alt:.0f} ft".format(
                        callsign=ac.get("callsign", "TFC"),
                        hdg=float(ac.get("heading", 0.0)),
                        spd=float(ac.get("speed", 0.0)),
                        alt=float(ac.get("altitude", 0.0)),
                    )
                )

        return "\n".join(lines)

    def _coerce_action(self, action: Union[ATCAction, str]) -> ATCAction:
        if isinstance(action, ATCAction):
            return action

        action_text = str(action).strip()
        match = self.CLEARANCE_PATTERN.match(action_text)
        if not match:
            return ATCAction(callsign="UNK", command="hold", value=0.0)

        callsign, command, value, runway = match.groups()
        numeric_value = float(value) if value is not None else None
        return ATCAction(
            callsign=callsign.upper(),
            command=command.lower(),
            value=numeric_value,
            runway=(runway.upper() if runway else None),
        )

    def _parse_clearance_to_gym_action(self, action: ATCAction) -> Any:
        space = self.gym_env.action_space

        ownship = self._extract_ownship(self.last_obs)
        current_hdg = float(ownship.get("heading", 180.0))
        current_spd = float(ownship.get("speed", 220.0))

        heading_delta = 0.0
        speed_delta = 0.0

        if action.command in {"heading", "vector"} and action.value is not None:
            target_hdg = float(action.value) % 360.0
            raw_delta = (target_hdg - current_hdg + 540.0) % 360.0 - 180.0
            heading_delta = float(np.clip(raw_delta / 30.0, -1.0, 1.0))

        if action.command == "speed" and action.value is not None:
            target_spd = float(action.value)
            raw_delta = target_spd - current_spd
            speed_delta = float(np.clip(raw_delta / 40.0, -1.0, 1.0))

        if action.command in {"land", "takeoff", "hold", "altitude"}:
            # MergeEnv-v0 generally controls heading/speed; unsupported commands become no-op or gentle slow-down.
            if action.command == "hold":
                speed_delta = -0.25

        #Different environments expose different action_space types.
        if isinstance(space, gym.spaces.Box): #Box means continuous vector controls (e.g., heading and speed deltas); we map the parsed clearance into this space accordingly.
            vec = np.zeros(space.shape, dtype=np.float32)
            flat = vec.reshape(-1)
            if flat.size >= 1:
                flat[0] = heading_delta
            if flat.size >= 2:
                flat[1] = speed_delta
            return flat.reshape(space.shape)

        if isinstance(space, gym.spaces.Discrete): #Discrete means a fixed set of discrete actions (e.g., turn left, turn right, speed up, slow down); we map the parsed clearance into the closest matching discrete action index. This is a conservative fallback and may not perfectly align with the intended clearance semantics.
            # Conservative discrete fallback mapping.
            if action.command in {"heading", "vector"}:
                if heading_delta < -0.2:
                    return min(0, space.n - 1)
                if heading_delta > 0.2:
                    return min(2, space.n - 1)
                return min(1, space.n - 1)
            if action.command == "speed":
                return min(1, space.n - 1)
            return 0

        if isinstance(space, gym.spaces.MultiDiscrete):
            vec = np.zeros_like(space.nvec)
            return vec

        return 0

    def _compute_grader_reward(
        self,
        gym_reward: float,
        conflicts: int,
        done: bool,
        action: ATCAction,
        progress_score: float,
        track_error_deg: float,
    ) -> Tuple[float, Dict[str, float]]:
        compliance = self._compliance_score(action)
        safety = 1.0 if conflicts == 0 else max(0.0, 1.0 - 0.35 * conflicts)

        step_idx = max(1, int(self.current_scenario["step"]))
        max_steps = max(1, int(self.current_scenario["max_steps"]))
        step_progress = step_idx / max_steps
        track_alignment = float(np.clip(1.0 - (track_error_deg / 90.0), 0.0, 1.0))
        efficiency = float(np.clip(0.55 * progress_score + 0.45 * (1.0 - step_progress), 0.0, 1.0))

        normalized_env = float(np.tanh(gym_reward))
        goal = float(np.clip((normalized_env + 1.0) / 2.0, 0.0, 1.0)) #basically maps gym reward to [0,1] range with diminishing returns for large positive rewards; this allows the GRPO reward to reflect both the dense gym reward signal and the sparse goal completion signal in a balanced way.
        if done and conflicts == 0: #if terminal and no conflicts, add difficulty bonus.
            diff = self.current_scenario["difficulty"]
            goal += float(self.difficulty_levels[diff]["goal_bonus"])
            goal = float(np.clip(goal, 0.0, 1.0))

        # Weighted shape used for GRPO: dense, bounded, and interpretable.
        reward = (
            0.35 * safety
            + 0.25 * efficiency
            + 0.20 * compliance
            + 0.10 * track_alignment
            + 0.20 * goal
        )

        reward_parts = {
            "safety": safety,
            "efficiency": efficiency,
            "compliance": compliance,
            "track_alignment": track_alignment,
            "progress_score": progress_score,
            "goal": goal,
            "gym": gym_reward,
        }
        return reward, reward_parts

    def _agent_graders(self, done: bool, conflicts: int, reward_parts: Dict[str, float]) -> Dict[str, float]:
        # Hackathon requirement: 3 task graders with 0.0-1.0 scores (easy -> medium -> hard).
        easy_safety = float(np.clip(1.0 - 0.30 * conflicts, 0.0, 1.0)) #Easy task should mostly test safety
        medium_efficiency = float(np.clip(0.5 * reward_parts["efficiency"] + 0.5 * reward_parts["goal"], 0.0, 1.0)) #Medium task should test efficiency and progress towards goal
        hard_phraseology = float(np.clip(0.5 * reward_parts["compliance"] + 0.5 * easy_safety, 0.0, 1.0)) #Hard task should test compliance and safety

        overall = float(np.clip((easy_safety + medium_efficiency + hard_phraseology) / 3.0, 0.0, 1.0))
        if done and conflicts > 0:
            overall *= 0.8

        return {
            "task_easy_safety": easy_safety,
            "task_medium_efficiency": medium_efficiency,
            "task_hard_phraseology": hard_phraseology,
            "overall": overall,
        }

    def _compute_progress_score(self, distance_to_faf: Optional[float]) -> float:
        if distance_to_faf is None:
            return 0.5

        if self.prev_distance_to_faf is None:
            self.prev_distance_to_faf = distance_to_faf
            return 0.5

        delta = self.prev_distance_to_faf - distance_to_faf
        self.prev_distance_to_faf = distance_to_faf

        # Positive when ownship gets closer to FAF, negative when diverging.
        return float(np.clip(0.5 + (delta / 10.0), 0.0, 1.0))

    def _extract_distance_to_faf(self, obs: Dict[str, Any]) -> Optional[float]:
        candidate_keys = [
            "distance_to_faf",
            "faf_distance",
            "dist_to_faf",
            "distance_to_waypoint",
            "waypoint_distance",
            "wp_dist",
        ]
        for key in candidate_keys:
            value = obs.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0 and isinstance(value[0], (int, float)):
                return float(value[0])
        return None

    def _extract_track_error_deg(self, obs: Dict[str, Any]) -> float:
        own_hdg = self._first_number(obs, ["heading", "hdg", "track"], default=180.0)
        target_bearing = self._first_number(
            obs,
            ["bearing_to_faf", "faf_bearing", "bearing_to_waypoint", "waypoint_bearing", "wp_bearing"],
            default=own_hdg,
        )
        return float(abs((target_bearing - own_hdg + 540.0) % 360.0 - 180.0))

    def _compliance_score(self, action: ATCAction) -> float:
        if not action.callsign or not action.command:
            return 0.0

        allowed = {"heading", "speed", "altitude", "takeoff", "land", "hold", "vector"}
        if action.command not in allowed:
            return 0.0

        score = 0.6
        if action.command in {"heading", "speed", "altitude", "vector"}:
            score += 0.2 if action.value is not None else 0.0
        if action.command in {"takeoff", "land"}:
            score += 0.2 if action.runway else 0.0
        if action.command == "hold":
            score += 0.2

        return float(np.clip(score, 0.0, 1.0))

    def _extract_ownship(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if "ownship" in obs and isinstance(obs["ownship"], dict):
            return obs["ownship"]

        return {
            "callsign": obs.get("callsign", "OWN"),
            "heading": self._first_number(obs, ["heading", "hdg", "track"], default=180.0),
            "speed": self._first_number(obs, ["speed", "tas", "gs"], default=220.0),
            "altitude": self._first_number(obs, ["altitude", "alt", "z"], default=10000.0),
        }

    def _extract_traffic(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "traffic" in obs and isinstance(obs["traffic"], list):
            return [ac for ac in obs["traffic"] if isinstance(ac, dict)]

        aircraft = self.last_info.get("aircraft")
        if isinstance(aircraft, list):
            return [ac for ac in aircraft if isinstance(ac, dict)]

        return []

    def _active_aircraft_count(self, obs: Dict[str, Any], info: Dict[str, Any]) -> int:
        traffic = self._extract_traffic(obs)
        if traffic:
            return len(traffic)

        aircraft = info.get("aircraft") if isinstance(info, dict) else None
        if isinstance(aircraft, list):
            return len(aircraft)

        n_ac = self.difficulty_levels[self.current_scenario.get("difficulty", "medium")]["n_ac"]
        return int(n_ac)

    def _format_action(self, action: ATCAction) -> str:
        parts = [f"{action.callsign}", action.command]
        if action.value is not None:
            value_text = f"{action.value:.0f}" if float(action.value).is_integer() else f"{action.value:.1f}"
            parts.append(value_text)
        if action.runway:
            parts.extend(["runway", action.runway])
        return "[clear " + " ".join(parts).strip() + "]"

    @staticmethod
    def _first_number(obs: Dict[str, Any], keys: List[str], default: float) -> float:
        for key in keys:
            value = obs.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, (list, tuple, np.ndarray)) and value:
                first = value[0]
                if isinstance(first, (int, float)):
                    return float(first)
        return default

    @staticmethod
    def _obs_to_dict(obs: Any) -> Dict[str, Any]:
        def _to_builtin(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): _to_builtin(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_builtin(v) for v in value]
            if isinstance(value, np.ndarray):
                return [_to_builtin(v) for v in value.tolist()]
            if isinstance(value, np.generic):
                return value.item()
            return value

        if isinstance(obs, dict):
            return _to_builtin(dict(obs))

        if isinstance(obs, np.ndarray):
            return {"vector": _to_builtin(obs)}

        if hasattr(obs, "__dict__"):
            try:
                return _to_builtin(dict(vars(obs)))
            except Exception:
                pass

        return {"raw": str(obs)}


def run_baseline_episode(seed: int = 7, difficulty: str = "medium") -> Dict[str, float]:
    """Deterministic baseline rollout for reproducible hackathon scoring."""

    env = ATCAdvisorEnv(default_difficulty=difficulty, seed=seed)
    obs = env.reset(seed=seed)

    done = False
    step_count = 0
    grader_overall = 0.0

    while not done and step_count < env.current_scenario["max_steps"]:
        # Simple baseline: keep converging to a conservative merge profile.
        action = ATCAction(callsign="OWN", command="heading", value=180.0)
        obs, reward, done, info = env.step(action)
        grader_overall = float(info["grader_scores"]["overall"])
        step_count += 1

        if step_count % 4 == 0 and not done:
            obs, reward, done, info = env.step(ATCAction(callsign="OWN", command="speed", value=210.0))
            grader_overall = float(info["grader_scores"]["overall"])
            step_count += 1

    result = {
        "difficulty": difficulty,
        "steps": float(step_count),
        "overall_grader": float(np.clip(grader_overall, 0.0, 1.0)),
        "cumulative_reward": float(env.cumulative_reward),
        "conflicts": float(obs.conflicts),
    }

    env.close()
    return result


def extract_clearance(completion_text: str) -> ATCAction:
    """Parse model text into a structured ATCAction using bracketed clearance format."""

    pattern = re.compile(
        r"\[\s*clear\s+([A-Za-z0-9\-]+)\s+([a-zA-Z]+)(?:\s+([\-]?[0-9]+(?:\.[0-9]+)?))?(?:\s+runway\s+([A-Za-z0-9]+))?.*\]",
        re.IGNORECASE,
    )
    match = pattern.search(completion_text or "")
    if not match:
        return ATCAction(callsign="OWN", command="hold", value=0.0)

    callsign, command, value, runway = match.groups()
    return ATCAction(
        callsign=(callsign or "OWN").upper(),
        command=(command or "hold").lower(),
        value=(float(value) if value is not None else None),
        runway=(runway.upper() if runway else None),
    )


def format_atc_history(messages: List[Dict[str, str]]) -> str:
    """Format ATC message history for prompt context."""

    lines: List[str] = []
    for message in messages:
        if isinstance(message, dict):
            category = str(message.get("category", "MESSAGE"))
            content = str(message.get("content", "")).strip()
        else:
            category = str(getattr(message, "category", "MESSAGE"))
            content = str(getattr(message, "content", "")).strip()

        if content:
            lines.append(f"[{category}] {content}")
    return "\n".join(lines)


def make_atc_user_prompt(prompt_text: str, observation: ATCObservation) -> str:
    """Construct the user prompt payload passed to the model each turn."""

    base_prompt = prompt_text.strip() if prompt_text and prompt_text.strip() else "ATC merge advisory task"
    history = format_atc_history(observation.messages)
    history_section = history if history else "[SYSTEM] No prior clearances."

    return (
        f"Task:\n{base_prompt}\n\n"
        f"Radar snapshot:\n{observation.radar_text}\n\n"
        f"Conversation so far:\n{history_section}\n\n"
        "Respond with one clearance in this format only: "
        "[clear CALLSIGN COMMAND VALUE] or [clear CALLSIGN land runway 27L]."
    )


def rollout_once(
    trainer: Any,
    env: ATCAdvisorEnv,
    tokenizer: Any,
    dataset_prompt: str,
    system_prompt: str,
    max_turns: int = 20,
) -> Dict[str, Any]:
    """
    Execute one full ATC episode for GRPO, following the same structure as 04-training.md.
    """

    from trl.experimental.openenv import generate_rollout_completions

    observation = env.reset()
    done = False

    prompt_ids: List[int] = []
    completion_ids: List[int] = []
    logprobs: List[float] = []
    raw_rewards: List[float] = []
    safety_scores: List[float] = []
    efficiency_scores: List[float] = []
    compliance_scores: List[float] = []
    final_goal_scores: List[float] = []

    for _turn in range(max_turns):
        if done:
            break

        user_prompt = make_atc_user_prompt(dataset_prompt, observation)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs.get("prompt_ids", []))
        completion_ids.extend(rollout_outputs.get("completion_ids", []))
        logprobs.extend(rollout_outputs.get("logprobs", []))

        completion_text = rollout_outputs.get("text")
        if not completion_text:
            completion_text = tokenizer.decode(
                rollout_outputs.get("completion_ids", []),
                skip_special_tokens=True,
            )

        action = extract_clearance(completion_text)
        observation, reward, done, info = env.step(action)

        raw_rewards.append(float(reward or 0.0))

        reward_parts = info.get("reward_parts", {}) if isinstance(info, dict) else {}
        grader_scores = info.get("grader_scores", {}) if isinstance(info, dict) else {}

        safety_scores.append(float(grader_scores.get("task_easy_safety", reward_parts.get("safety", 0.0))))
        efficiency_scores.append(
            float(grader_scores.get("task_medium_efficiency", reward_parts.get("efficiency", 0.0)))
        )
        compliance_scores.append(
            float(grader_scores.get("task_hard_phraseology", reward_parts.get("compliance", 0.0)))
        )
        final_goal_scores.append(float(reward_parts.get("goal", 0.0)))

    final_goal = final_goal_scores[-1] if final_goal_scores else 0.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "raw_rewards": raw_rewards,
        "safety_reward": safety_scores[-1] if safety_scores else 0.0,
        "efficiency_reward": efficiency_scores[-1] if efficiency_scores else 0.0,
        "compliance_reward": compliance_scores[-1] if compliance_scores else 0.0,
        "final_goal_reward": final_goal,
        # Alias for older trainer setups that expect a single "correct" terminal score.
        "correct_reward": final_goal,
    }


def rollout_func(
    prompts: List[str],
    trainer: Any = None,
    env: Optional[ATCAdvisorEnv] = None,
    tokenizer: Any = None,
    system_prompt: str = "",
    max_turns: int = 20,
) -> Dict[str, List[Any]]:
    """Batch rollout wrapper matching the TRL GRPO rollout_func interface."""

    if env is None:
        raise ValueError("rollout_func requires an initialized ATCAdvisorEnv via env=...")
    if tokenizer is None:
        raise ValueError("rollout_func requires a tokenizer via tokenizer=...")

    episode_prompt_ids: List[List[int]] = []
    episode_completion_ids: List[List[int]] = []
    episode_logprobs: List[List[float]] = []
    safety_rewards: List[float] = []
    efficiency_rewards: List[float] = []
    compliance_rewards: List[float] = []
    final_goal_rewards: List[float] = []
    correct_rewards: List[float] = []

    for prompt_text in prompts:
        episode = rollout_once(
            trainer=trainer,
            env=env,
            tokenizer=tokenizer,
            dataset_prompt=prompt_text,
            system_prompt=system_prompt,
            max_turns=max_turns,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        safety_rewards.append(float(episode["safety_reward"]))
        efficiency_rewards.append(float(episode["efficiency_reward"]))
        compliance_rewards.append(float(episode["compliance_reward"]))
        final_goal_rewards.append(float(episode["final_goal_reward"]))
        correct_rewards.append(float(episode["correct_reward"]))

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "safety_reward": safety_rewards,
        "efficiency_reward": efficiency_rewards,
        "compliance_reward": compliance_rewards,
        "final_goal_reward": final_goal_rewards,
        "correct_reward": correct_rewards,
    }


def reward_safety(completions: List[Any], **kwargs: Any) -> List[float]:
    rewards = kwargs.get("safety_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_efficiency(completions: List[Any], **kwargs: Any) -> List[float]:
    rewards = kwargs.get("efficiency_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_compliance(completions: List[Any], **kwargs: Any) -> List[float]:
    rewards = kwargs.get("compliance_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_final_goal(completions: List[Any], **kwargs: Any) -> List[float]:
    rewards = kwargs.get("final_goal_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


if __name__ == "__main__":
    summary = run_baseline_episode(seed=7, difficulty="medium")
    print("Baseline rollout summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
