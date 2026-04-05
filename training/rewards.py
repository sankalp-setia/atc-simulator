from typing import Any, List


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
