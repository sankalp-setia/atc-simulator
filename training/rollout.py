from typing import Any, Dict, List

from atc_advisor_env import ATCAdvisorEnv, rollout_once


def rollout_func(
    prompts: List[str],
    trainer: Any = None,
    env: ATCAdvisorEnv = None,
    tokenizer: Any = None,
    system_prompt: str = "",
    max_turns: int = 20,
) -> Dict[str, List[Any]]:
    if env is None:
        raise ValueError("rollout_func requires env=ATCAdvisorEnv(...)")
    if tokenizer is None:
        raise ValueError("rollout_func requires tokenizer=...")

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
