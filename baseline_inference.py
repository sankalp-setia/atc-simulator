from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from atc_advisor_env import ATCAdvisorEnv, extract_clearance


DEFAULT_SYSTEM_PROMPT = """
You are an expert AI co-pilot assisting human air traffic controllers.
Your job is to issue safe, efficient, and compliant clearances for merging traffic on final approach.

Rules:
- Always prioritize separation and conflict avoidance.
- Use one clearance per turn.
- Format exactly: [clear CALLSIGN COMMAND VALUE]
- Example: [clear OWN heading 180]
""".strip()


TASKS: List[Dict[str, Any]] = [
    {
        "id": "task_easy_safety",
        "difficulty": "easy",
        "objective": "Maintain zero conflicts for light traffic merge operations.",
    },
    {
        "id": "task_medium_efficiency",
        "difficulty": "medium",
        "objective": "Balance safe separation and efficient progression to merge profile.",
    },
    {
        "id": "task_hard_phraseology",
        "difficulty": "hard",
        "objective": "Maintain safety under dense traffic while preserving command compliance.",
    },
]


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text or "[clear OWN hold]"


class VLLMBackend(LLMBackend):
    def __init__(self, model: str):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm is not installed. Install with: pip install vllm") from exc

        self.llm = LLM(model=model)
        self.sampling = SamplingParams(temperature=0.0, max_tokens=64)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        outputs = self.llm.generate([prompt], self.sampling)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return text or "[clear OWN hold]"


class TransformersBackend(LLMBackend):
    def __init__(self, model: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=64,
            do_sample=False,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return text or "[clear OWN hold]"


class HeuristicBackend(LLMBackend):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt, user_prompt
        return "[clear OWN heading 180]"


def get_backend(provider: str, model: str) -> LLMBackend:
    p = provider.lower()
    if p == "openai":
        return OpenAIBackend(model=model)
    if p == "vllm":
        return VLLMBackend(model=model)
    if p in {"hf", "transformers"}:
        return TransformersBackend(model=model)
    if p == "heuristic":
        return HeuristicBackend()
    raise ValueError(f"Unsupported provider: {provider}")


def build_user_prompt(observation_text: str, turn: int) -> str:
    return (
        f"Turn {turn}.\n"
        "Issue one ATC clearance in bracket format.\n"
        f"Radar:\n{observation_text}\n"
    )


def run_task(
    backend: LLMBackend,
    task: Dict[str, Any],
    seed: int = 7,
    max_turns: int = 20,
) -> Dict[str, float]:
    env = ATCAdvisorEnv(default_difficulty=task["difficulty"], seed=seed)
    obs = env.reset(seed=seed)

    done = False
    turn = 0
    final_scores: Dict[str, float] = {}

    while not done and turn < max_turns:
        user_prompt = build_user_prompt(obs.radar_text, turn + 1)
        output_text = backend.generate(DEFAULT_SYSTEM_PROMPT, user_prompt)
        action = extract_clearance(output_text)

        obs, _reward, done, info = env.step(action)
        final_scores = info.get("grader_scores", {}) if isinstance(info, dict) else {}
        turn += 1

    env.close()

    return {
        "task_score": float(final_scores.get("overall", 0.0)),
        "easy": float(final_scores.get("task_easy_safety", 0.0)),
        "medium": float(final_scores.get("task_medium_efficiency", 0.0)),
        "hard": float(final_scores.get("task_hard_phraseology", 0.0)),
        "turns": float(turn),
    }


def run_baseline(
    model: str = "Qwen/Qwen3-1.7B",
    seed: int = 7,
    provider: str = "vllm",
    max_turns: int = 20,
) -> Dict[str, Any]:
    backend = get_backend(provider=provider, model=model)

    task_results: Dict[str, Dict[str, float]] = {}
    for task in TASKS:
        task_results[task["id"]] = run_task(
            backend=backend,
            task=task,
            seed=seed,
            max_turns=max_turns,
        )

    overall = sum(v["task_score"] for v in task_results.values()) / max(len(task_results), 1)

    return {
        "provider": provider,
        "model": model,
        "seed": seed,
        "max_turns": max_turns,
        "overall_score": float(overall),
        "tasks": task_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline evaluation for atc-advisor-v0")
    parser.add_argument("--provider", type=str, default="heuristic", choices=["openai", "vllm", "transformers", "hf", "heuristic"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_baseline(
        provider=args.provider,
        model=args.model,
        seed=args.seed,
        max_turns=args.max_turns,
    )
    print(json.dumps(result, indent=2))
