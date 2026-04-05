from __future__ import annotations

import argparse
from functools import partial
from typing import Any, Dict

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from atc_advisor_env import ATCAdvisorEnv
from training.rollout import rollout_func
from training.rewards import reward_compliance, reward_efficiency, reward_final_goal, reward_safety


SYSTEM_PROMPT = """
You are an expert AI co-pilot assisting human air traffic controllers.
Your job is to issue safe, efficient, and compliant clearances for merging traffic on final approach.

Rules:
- Always prioritize separation and conflict avoidance.
- Use standard phraseology.
- Issue only one clearance per turn.
- Reply in bracket format only.
- Format: [clear CALLSIGN COMMAND VALUE] or [clear CALLSIGN land runway 27L]
""".strip()


def build_dataset(prompt_text: str, dataset_size: int) -> Dataset:
    return Dataset.from_dict({"prompt": [prompt_text] * dataset_size})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GRPO runner for atc-advisor-v0")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="HF model id")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--output-dir", type=str, default="outputs/atc-grpo")
    parser.add_argument("--seed", type=int, default=7) 
    parser.add_argument(
        "--prompt",
        type=str,
        default="Act as an ATC merge advisor and issue one safe clearance each turn.",
    )
    parser.add_argument("--use-vllm", action="store_true", help="Enable vLLM generation inside TRL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    env = ATCAdvisorEnv(default_difficulty=args.difficulty, seed=args.seed)

    dataset = build_dataset(args.prompt, args.dataset_size)

    # Bind env/tokenizer/system prompt so GRPO can call rollout_func(prompts, trainer=...).
    bound_rollout = partial(
        rollout_func,
        env=env,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_turns=args.max_turns,
    )

    config_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "save_steps": 20,
        "max_completion_length": 64,
        "num_generations": args.num_generations,
        "use_vllm": args.use_vllm,
        "report_to": [],
        "seed": args.seed,
    }
    if args.use_vllm:
        config_kwargs["vllm_mode"] = "colocate"

    grpo_config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_safety,
            reward_efficiency,
            reward_compliance,
            reward_final_goal,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=bound_rollout,
    )

    try:
        trainer.train()
        trainer.save_model(args.output_dir)
        print(f"Saved GRPO model to {args.output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
