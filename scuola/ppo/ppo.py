import os
import time
import gc
import re
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, ShardingStrategy,
    StateDictType, FullStateDictConfig,
)
import numpy as np
from tqdm import trange

from datasets import load_dataset, Dataset

# For inference
from vllm import LLM, SamplingParams

# For huggingface model/tokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# --------------
#  Our utilities
# --------------
from scuola.utils import (
    prepare_model_inputs, compute_token_log_probs, evaluate_on_test_set,
    dump_episodes, load_model_into_vllm,
    mlflow_initialize, mlflow_log_params, mlflow_end_run, mlflow_log_metrics
)
from scuola.config import (
    ModelConfig, TokenizerConfig, FsdpConfig, VllmConfig, SchedulerConfig,
    OptimizerConfig, DatasetConfig, TrainLoaderConfig, MlflowConfig,
    LoggersConfig, PPOConfig, Config, load_and_validate_config
)

###############################################################################
# Logging
###############################################################################
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

###############################################################################
# PPO Reward Functions
###############################################################################

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process in the mind "
    "and then provide the user with the answer."
)

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final equation and answer in "
    "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
)

def format_reward_func(completion: str, eos_token: str) -> float:
    """
    Checks if completion has <think>...</think> and <answer>...</answer> in correct format,
    and if the <answer> content only has digits and + - * / ( ) .
    """
    if not eos_token:
        raise ValueError(f"eos_token cannot be empty")

    allowed_pattern = r"^[\d+\-*/().\s]+$"
    try:
        # Some lines add <think> at the start
        #completion = "<think>" + completion

        # Remove EOS if present
        if completion.endswith(eos_token):
            completion = completion[: -len(eos_token)]

        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)
        if not match or len(match.groups()) != 2:
            return 0.0

        answer_content = match.group(2).strip()
        if not re.match(allowed_pattern, answer_content):
            return 0.5
        return 1.0

    except Exception as e:
        return 0.0

def equation_reward_func(completion: str, nums: list[int], target: int) -> float:
    """
    Checks if the final <answer> expression is mathematically correct, uses all numbers exactly once, etc.
    """
    try:
        completion = "<think>" + completion
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        equation = match.group(1).strip()

        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
        if sorted(used_numbers) != sorted(nums):
            return 0.0

        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate with no builtins
        result = eval(equation, {"__builtins__": None}, {})
        return 1.0 if abs(float(result) - float(target)) < 1e-5 else 0.0
    except:
        return 0.0

def compute_reward(completion: str, sample: dict[str, list], eos_token: str) -> Tuple[float, dict[str, float]]:
    """
    Total reward = format_reward + equation_reward
    """
    nums = sample["nums"]
    target = sample["target"]

    fr = format_reward_func(completion, eos_token)
    er = equation_reward_func(completion, nums, target)

    reward = fr + er
    return reward, {"format_reward": fr, "equation_reward": er}

###############################################################################
# PPO Logic
###############################################################################

def create_training_episodes(
    samples: Dataset,
    all_generations: list[list[int]],
    all_finish_reasons: list[str],
    tokenizer: AutoTokenizer,
    eos_token_id: int,
    eos_token: str,
    generations_per_sample: int,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    """
    Group the generations by sample, compute rewards, compute normalized advantages.
    """
    assert len(all_generations) == len(samples) * generations_per_sample, f"Got {len(all_generations)=} vs {len(samples)=} x {generations_per_sample=}"
    assert len(all_generations) == len(all_finish_reasons), f"But got {len(all_generations)=} vs {len(all_finished_reasons)=}"

    # Group indices
    groups = [
        list(range(i, i + generations_per_sample))
        for i in range(0, len(all_generations), generations_per_sample)
    ]

    all_query_token_ids = []
    all_responses_token_ids = []
    all_advantages = []

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    # Loop over each input sample
    for sample, group_indices in zip(samples, groups):
        # Gather the generations for this sample
        response_token_ids_group = [all_generations[g_idx] for g_idx in group_indices]
        finish_reasons_group = [all_finish_reasons[g_idx] for g_idx in group_indices]

        # Convert to text
        responses = tokenizer.batch_decode(response_token_ids_group, skip_special_tokens=False)

        # Compute rewards
        rewards_and_metrics = [compute_reward(resp, sample, eos_token) for resp in responses]
        rewards, reward_metrics_list = zip(*rewards_and_metrics)

        rewards_np = np.array(rewards)
        advantages_np = (rewards_np - rewards_np.mean()) / (rewards_np.std() + 1e-4)

        log.info(f"Advantage mean: {advantages_np.mean()}, std: {advantages_np.std()}")

        per_token_adv = [
            [adv] * len(r_ids)
            for adv, r_ids in zip(advantages_np, response_token_ids_group)
        ]

        # Save
        all_query_token_ids.extend([sample["input_ids"]] * generations_per_sample)
        all_responses_token_ids.extend(response_token_ids_group)
        all_advantages.extend(per_token_adv)

        stats["rewards"].extend(rewards_np)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons_group])
        stats["response_lengths"].extend([len(r_ids) for r_ids in response_token_ids_group])

        # Log format / equation reward
        for rm_dict in reward_metrics_list:
            for k, v in rm_dict.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    # Combine episodes
    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }
    return episodes, stats


def compute_pg_loss(
    policy_model: nn.Module,
    reference_model: nn.Module,
    batch: dict[str, torch.Tensor],
    total_response_len: int,
    temperature: float,
    kl_coefficient: float,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """
    1. log_probs for policy_model & reference_model
    2. KL penalty
    3. policy gradient with advantages
    4. combine
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    advantages = batch["advantages"]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    labels_mask = (labels[..., 1:] != -100).float()

    # Reference log probs (no gradient)
    with torch.no_grad():
        ref_logps = compute_token_log_probs(reference_model, model_inputs, temperature=temperature)

    # Policy log probs
    logps = compute_token_log_probs(policy_model, model_inputs, temperature=temperature)

    # KL penalty
    #   kl = exp(ref - policy) - (ref - policy) - 1
    #   => = exp( delta ) - delta - 1
    #     with delta = ref_logps - policy_logps
    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1.0
    kl_penalty = kl_penalty * labels_mask

    # Entropy
    entropy = -logps.sum() / labels_mask.sum()

    # Policy gradient loss = - log(p) * advantage
    # SHIFT advantage by 1 token as well
    advantages_shifted = advantages[..., 1:]
    policy_loss = -logps * advantages_shifted
    policy_loss = policy_loss * labels_mask

    # Combine
    loss = (policy_loss + kl_coefficient * kl_penalty).sum() / total_response_len

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len,
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "entropy": entropy.item(),
    }
    return loss, metrics

###############################################################################
# Data Preparation
###############################################################################

def preprocess_example(
    example: dict[str, Any],
    tokenizer: AutoTokenizer,
):
    """
    Turn {nums, target} into input_ids for the LLM.
    """
    numbers = example["nums"]
    target = example["target"]

    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    # We'll rely on a custom "apply_chat_template" approach. If your tokenizer doesn't have it,
    # just manually build a string and tokenize via tokenizer(...)
    # For demonstration, we assume "apply_chat_template(..., tokenize=True)":
    input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)
    return {
        "prompt": tokenizer.decode(input_ids, skip_special_tokens=False),
        "input_ids": input_ids,
        "nums": numbers,
        "target": target,
    }

def data_prep(
    tokenizer: AutoTokenizer,
    cfg: DatasetConfig,
    num_train_samples: int = 2000,
    num_test: int = 500,
):
    """
    Example dataset from huggingface.
    Splits into train/test.
    """
    dataset = load_dataset(cfg.dataset_name, split="train")
    dataset = dataset.map(
        lambda ex: preprocess_example(ex, tokenizer=tokenizer),
        num_proc=4,
    )
    dataset = dataset.take(num_train_samples + num_test)

    # Train/test split
    train_test_split = dataset.train_test_split(test_size=num_test, seed=42)
    return train_test_split["train"], train_test_split["test"]

###############################################################################
# Main
###############################################################################


def init_distributed():
    """
    Initialize torch.distributed via environment variables from torchrun.
    """
    if not dist.is_available():
        raise RuntimeError("Distributed not available!")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=torch.distributed.constants.default_pg_timeout)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def build_model_and_tokenizer(tokenizer_cfg: TokenizerConfig,
                              model_cfg: ModelConfig):
    """
    Build HF model and tokenizer in pure PyTorch, wrap with FSDP.
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg.name, trust_remote_code=True, padding_side="left")

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.pretrained_model_name_or_path,
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
        torch_dtype = torch.bfloat16
    )

    # Freeze the reference model by copy
    import copy
    reference_model = copy.deepcopy(model)
    reference_model.eval()

    return model, reference_model, tokenizer


def wrap_fsdp(model: nn.Module, fsdp_cfg: FsdpConfig):
    """
    Wrap model with PyTorch FSDP.
    """
    # Convert user string to an actual ShardingStrategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    sharding = strategy_map.get(fsdp_cfg.sharding_strategy.upper(), ShardingStrategy.FULL_SHARD)

    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding,
        # For real usage, see the many FSDP configs, e.g. CPU offload, etc.
        device_id=torch.cuda.current_device(),
    )

    # Make sure we can get a full state dict
    FSDP.set_state_dict_type(
        fsdp_model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(),
    )
    return fsdp_model


def build_optimizer(model: nn.Module, optimizer_cfg: OptimizerConfig):
    """
    Simple AdamW optimizer for the policy model.
    """
    optimizer = optim.AdamW(model.parameters(), lr=optimizer_cfg.lr)
    return optimizer


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    # Load config from YAML or use defaults
    if args.config_path:
        log.info(f"Loading config from {args.config_path}")
        try:
            cfg = load_and_validate_config(args.config_path)
        except Exception as e:
            raise ValueError(f"Error loading configuration: {args.config_path}") from e
    else:
        # Fallback
        log.info("No config path specified. Using default Config.")
        cfg = Config()

    log.info(f"read {cfg!r}")

    # Initialize distributed
    local_rank = init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Seeding
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    # Build model & tokenizer
    policy_model, reference_model, tokenizer = build_model_and_tokenizer(cfg.tokenizer, cfg.model)

    if cfg.mixed_precision:
        policy_model = policy_model.half()
        reference_model = reference_model.half()

    # Wrap them in FSDP
    fsdp_policy = wrap_fsdp(policy_model, cfg.fsdp)
    fsdp_reference = wrap_fsdp(reference_model, cfg.fsdp)

    fsdp_reference.eval()

    # Build optimizer
    optimizer = build_optimizer(fsdp_policy, cfg.optimizer)

    # Load dataset
    train_dataset, test_dataset = data_prep(tokenizer, cfg.train_loader.dataset)

    # Figure out microbatching
    per_device_batch_size = max(1, cfg.global_train_batch_size // world_size)
    device_microbatch_size = min(per_device_batch_size, cfg.device_train_microbatch_size)
    accumulation_steps = max(1, per_device_batch_size // device_microbatch_size)

    # Set up vLLM for inference
    # (If you want multi-GPU inference, set `tensor_parallel_size=cfg.inference_tp_size` etc.)
    inference_engine = LLM(
        model=cfg.model.pretrained_model_name_or_path,
        trust_remote_code=True,
        enforce_eager=True,
        dtype=torch.bfloat16,
        tensor_parallel_size=cfg.vllm.inference_tp_size,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        enable_sleep_mode=False,
        disable_log_stats=True,
        skip_tokenizer_init=True,
        seed=cfg.seed,
    )

    # EOS details
    eos_token_id = tokenizer.eos_token_id
    eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)

    # (Optional) init MLflow only on rank 0
    mlflow_initialize(cfg.loggers.mlflow)
    mlflow_log_params(cfg)

    # Optionally load weights into vLLM once at start
    dist.barrier()
    load_model_into_vllm(fsdp_policy, inference_engine)
    dist.barrier()

    begin_iter = 0

    # Main training loop of PPO
    cfg = cfg.ppo

    for iteration in trange(begin_iter, cfg.num_iterations, disable=(rank != 0)):
        # Evaluate every N steps
        if iteration % cfg.eval_interval == 0:
            log.info(f"Rank0: Evaluating at iteration={iteration}")
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eos_token=eos_token,
                eval_sampling_params=SamplingParams(
                    temperature=0.3,
                    max_tokens=1024,
                    n=1,
                    detokenize=False,
                    stop_token_ids=[eos_token_id],
                ),
                reward_func=lambda c, s: compute_reward(c, s, eos_token),
                local_rank = local_rank,
            )
            # Dump some example lines
            if rank == 0:
                dump_episodes(None, eval_episodes, eval_stats, tokenizer, iteration, is_eval=True)
                # Log average reward to MLflow
                if "rewards" in eval_stats and len(eval_stats["rewards"]) > 0:
                    avg_eval_reward = float(np.mean(eval_stats["rewards"]))
                    mlflow_log_metrics({"eval/reward": avg_eval_reward}, step=iteration)

        # Sample training batch
        #   episodes_per_iteration => how many new episodes we gather each iteration
        #   generations_per_sample => how many completions per input
        np.random.seed(17)
        num_samples = cfg.episodes_per_iteration // cfg.generations_per_sample
        indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
        samples = train_dataset.select(indices)

        # Inference to get responses
        dist.barrier()
        gen_time = time.time()

        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=cfg.generations_per_sample,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                max_tokens=cfg.max_response_tokens,
                detokenize=False,
                stop_token_ids=[eos_token_id],
            ),
        )
        dist.barrier()

        # Flatten out the generation results
        all_generations = []
        all_finish_reasons = []
        for out in outputs:
            for gen in out.outputs:
                all_generations.append(list(gen.token_ids))
                all_finish_reasons.append(gen.finish_reason)

        log.info(f"Rank {rank} generated {len(all_generations)} responses in {time.time()-gen_time:.2f}s")

        # Build episodes
        episodes, episodes_stats = create_training_episodes(
            samples,
            all_generations,
            all_finish_reasons,
            tokenizer,
            eos_token_id,
            eos_token,
            cfg.generations_per_sample,
        )

        # Dump a couple examples on rank0
        if rank == 0:
            dump_episodes(None, episodes, episodes_stats, tokenizer, iteration, is_eval=False)

        # Prepare model inputs for PPO training
        device = torch.device("cuda", local_rank)
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device=device,
        )

        fsdp_policy.train()
        fsdp_reference.eval()

        # total response tokens
        total_response_len = (model_inputs["labels"] != -100).sum().item()

        # Accumulate gradient
        metrics = {}
        batch_count = 0
        train_steps = len(model_inputs["input_ids"])

        for i in trange(0, train_steps, device_microbatch_size, disable=(rank != 0)):
            end_idx = min(i + device_microbatch_size, train_steps)

            batch = {
                k: v[i : end_idx] for k, v in model_inputs.items()
            }

            loss, loss_metrics = compute_pg_loss(
                policy_model=fsdp_policy,
                reference_model=fsdp_reference,
                batch=batch,
                total_response_len=total_response_len,
                temperature=cfg.temperature,
                kl_coefficient=cfg.kl_coeff,
            )

            # Track metrics in a local list
            for (k, val) in loss_metrics.items():
                metrics.setdefault(k, []).append(val)
            metrics.setdefault("loss", []).append(loss.item())

            # Backward on scaled loss
            batch_size_ratio = len(batch['input_ids']) / per_device_batch_size
            scaled_loss = loss * batch_size_ratio
            scaled_loss.backward()
            batch_count += 1

            #torch.nn.utils.clip_grad_norm_(fsdp_policy.parameters(), max_norm=1.0)

            # Step if we are at grad accumulation boundary
            if (batch_count % accumulation_steps == 0) or (end_idx >= train_steps):
                optimizer.step()
                optimizer.zero_grad()

            del loss, scaled_loss

        # Sync policy -> vLLM
        dist.barrier()
        inference_engine.wake_up()
        load_model_into_vllm(fsdp_policy, inference_engine)
        dist.barrier()

        # Summarize training metrics on rank0
        if rank == 0 and len(metrics["loss"]) > 0:
            train_metrics = {k: float(np.mean(vals)) for k, vals in metrics.items()}
            for k, v in train_metrics.items():
                mlflow_log_metrics({f"train/{k}": v}, step=iteration)

    # End training
    mlflow_end_run()
    dist.barrier()

if __name__ == "__main__":
    main()

