import os
from pathlib import Path

SCRATCH = Path.home() / "scratch"
os.environ["HF_HOME"] = str(SCRATCH / "hf_home")

import argparse
import gc
import re
import time
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, fields
from omegaconf import DictConfig

import numpy as np
import torch
import mlflow
from datasets import load_dataset
from tqdm import trange
from vllm import LLM, SamplingParams

from composer.checkpoint.load import load_checkpoint
from composer.utils import dist, get_device

from llmfoundry.utils.builders import (
    build_tokenizer,
    build_optimizer,
    build_logger,
    build_algorithm,
    build_composer_model,
)
from llmfoundry.utils.config_utils import (
    TrainConfig,
    process_init_device,
    update_batch_size_info,
    make_dataclass_and_log_config,
)
from llmfoundry.utils import (
    find_mosaicml_logger,
    maybe_create_mosaicml_logger,
)

from scuola.utils import (
    load_model_into_vllm,
    compute_token_log_probs,
    evaluate_on_test_set,
    dump_episodes,
    prepare_model_inputs,
)


############################################
# Prompts and Dataset
############################################

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

log = logging.getLogger(__name__)

@dataclass
class PPOTrainConfig(TrainConfig):
    vllm_config: Optional[dict[str, Any]] = None
    ppo_config: Optional[dict[str, Any]] = None

# Load and process dataset
def preprocess_example(
    example: Dict[str, Any],
    tokenizer: Any,
    SYSTEM_MESSAGE: str,
    PROMPT_TEMPLATE: str,
):
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    prefix = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    input_ids = tokenizer.apply_chat_template(prefix, tokenize=True, continue_final_message=True)
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"prompt": prompt, "input_ids": input_ids, "nums": numbers, "target": target}

def data_prep(tokenizer, llm_dataset):
    dataset = load_dataset(llm_dataset, split="train")
    dataset = dataset.map(
        preprocess_example,
        num_proc=6,
        fn_kwargs={
            "tokenizer": tokenizer,
            "SYSTEM_MESSAGE": SYSTEM_MESSAGE,
            "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        },
    )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset


def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    """
    Format: <think>...</think><answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Synthetically prepend <think> (if your pipeline relies on that to ease matching)
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(completion: str, sample: Dict[str, Any], EOS_TOKEN: str) -> Tuple[float, Dict[str, float]]:
    nums = sample["nums"]
    target = sample["target"]

    format_reward = format_reward_func(completion, EOS_TOKEN)
    equation_reward = equation_reward_func(completion=completion, nums=nums, target=target)

    reward = format_reward + equation_reward

    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }

    return reward, metrics


def create_training_episodes(
    samples: List[Dict[str, Any]],
    all_generations: List[List[int]],
    all_finish_reasons: List[str],
    tokenizer: Any,
    EOS_TOKEN_ID: int,
    EOS_TOKEN: str,
    GENERATIONS_PER_SAMPLE: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process model generations and calculate rewards for training episodes.

    This function processes generated responses and calculates rewards for training episodes by:
    1. Grouping generations by sample (GENERATIONS_PER_SAMPLE responses per input)
    2. Computing rewards and advantages for each response
    3. Processing response tokens (adding EOS tokens where needed)

    Args:
        samples: List of input samples, each containing:
            - input_ids: List[int], tokenized input prompt
            - nums: List[int], numbers to use in equation
            - target: int, target value for equation
        all_generations: List of token ID sequences for each generated response
        all_finish_reasons: List of finish reasons for each generation ("stop" or other)

    Returns:
        Tuple containing:
        1. Dictionary with processed data for training:
            - all_query_token_ids: List[List[int]], input token IDs repeated for each generation
            - all_response_token_ids: List[List[int]], response token IDs with EOS tokens added
            - all_advantages: List[List[float]], advantage values repeated for each token
        2. Dictionary with generation statistics:
            - response_lengths: List[int], lengths of generated responses
            - rewards: List[float], raw reward values
            - non_stop_rate: List[bool], whether each generation ended naturally
            - reward_metrics/*: Various reward component metrics
    """
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    # Process responses and calculate rewards
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE)) for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    for sample, group_indices in zip(samples, groups):
        response_token_ids = [all_generations[i] for i in group_indices]
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
        rewards_and_metrics = [compute_reward(resp, sample, EOS_TOKEN) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        per_token_advantages = [[adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)]

        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(per_token_advantages)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
        for rm, rm_dict in zip(reward_metrics, reward_metrics):
            for k, v in rm_dict.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes, stats


def compute_pg_loss(
    policy_model: Any,
    reference_model: Any,
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
    temperature: float,
    kl_coefficient: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss with KL penalty between policy and reference models.

    This function:
    1. Computes log probabilities for both policy and reference models
    2. Calculates KL divergence penalty between the models
    3. Computes policy gradient loss using advantages
    4. Combines the losses with KL coefficient

    Args:
        policy_model: The model being trained
        reference_model: The reference model for KL penalty calculation
        batch: Dictionary containing:
            - input_ids: Tensor of shape [batch_size, seq_len]
            - attention_mask: Tensor of shape [batch_size, seq_len]
            - labels: Tensor of shape [batch_size, seq_len] with -100 for ignored positions
            - advantages: Tensor of shape [batch_size, seq_len]

    Returns:
        Tuple containing:
            - loss: Combined policy gradient and KL penalty loss (scalar tensor)
            - metrics: Dictionary with detailed loss components:
                - policy_loss: Pure policy gradient loss
                - kl_penalty: KL divergence penalty
                - entropy: Policy entropy
    """
    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
    labels = batch["labels"]  # [batch_size, seq_len]
    advantages = batch["advantages"]  # [batch_size, seq_len]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    labels_mask = (labels[..., 1:] != -100).float()  # [batch_size, seq_len-1]

    with torch.no_grad():
        ref_logps = compute_token_log_probs(reference_model, model_inputs, ppo_cfg.temperature)  # [batch_size, seq_len-1]

    logps = compute_token_log_probs(policy_model, model_inputs, ppo_cfg.temperature)  # [batch_size, seq_len-1]

    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1  # [batch_size, seq_len-1]
    kl_penalty = kl_penalty * labels_mask  # [batch_size, seq_len-1]

    entropy = -logps.sum() / labels_mask.sum()  # scalar

    policy_loss = -logps * advantages[..., 1:]  # [batch_size, seq_len-1]
    policy_loss = policy_loss * labels_mask  # [batch_size, seq_len-1]

    loss = (policy_loss + ppo_cfg.kl_coeff * kl_penalty).sum() / total_response_len  # scalar

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len,
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "entropy": entropy.item(),
    }

    return loss, metrics


def main(cfg: DictConfig):

    TRAIN_CONFIG_KEYS = {field.name for field in fields(PPOTrainConfig)}

    logged_cfg, cfg = make_dataclass_and_log_config(
        cfg,
        PPOTrainConfig,
        TRAIN_CONFIG_KEYS,
        transforms='all',
    )

    # Get model configuration components
    fsdp_config: Optional[dict[str, Any]] = cfg.fsdp_config
    tp_config: Optional[dict[str, Any]] = cfg.tp_config
    vllm_config: Optional[dict[str, Any]] = cfg.vllm_config
    model_config: Optional[dict[str, Any]] = cfg.model

    model_name = model_config.get('name', "hf_causal_lm")
    per_device_batch_size = cfg.global_train_batch_size // dist.get_world_size()
    accumulation_steps = per_device_batch_size // cfg.device_train_microbatch_size

    # Set up distributed training
    device = get_device(None)
    dist.initialize_dist(device, timeout=cfg.dist_timeout)
    dist.barrier()

    # Build loggers
    loggers = [
        build_logger(str(name), logger_cfg)
        for name, logger_cfg in cfg.loggers.items()
    ] if cfg.loggers else []

    mosaicml_logger = find_mosaicml_logger(loggers)
    if mosaicml_logger is None:
        mosaicml_logger = maybe_create_mosaicml_logger()
        if mosaicml_logger is not None:
            loggers.append(mosaicml_logger)

    # Build tokenizer
    tokenizer = build_tokenizer(
        tokenizer_name=cfg.tokenizer.get('name', model_name),
        tokenizer_kwargs=cfg.tokenizer.get('kwargs', {})
    )

    # Initialize models
    init_context = process_init_device(cfg.model, fsdp_config, tp_config)

    # Build policy model with Composer
    name = model_config.pop('name')
    assert isinstance(name, str)
    print(type(model_config))
    assert isinstance(model_config, dict)
    policy_model = build_composer_model(
        name=model_name,
        tokenizer=tokenizer,
        init_context=init_context,
        master_weights_dtype=model_config.pop('master_weights_dtype', None),
        cfg=model_config
    )

    # Build reference model (frozen copy of policy model)
    reference_model = build_composer_model(
        name=model_name,
        tokenizer=tokenizer,
        init_context=init_context,
        master_weights_dtype=model_config.pop('master_weights_dtype', None),
        cfg=model_config
    )
    reference_model.eval()

    # Get tokenizer components
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    # Prepare datasets
    train_dataset, test_dataset = data_prep(tokenizer, "Jiayi-Pan/Countdown-Tasks-3to4")
    log.info(f"Train dataset size: {len(train_dataset)}")
    log.info(f"Test dataset size: {len(test_dataset)}")

    # Optimizer
    optimizer_name: str = cfg.optimizer.pop('name')
    optimizer = build_optimizer(policy_model, optimizer_name, cfg.optimizer)

    # Initialize vLLM for inference
    inference_engine = LLM(
        model=model_config.get('pretrained_model_name_or_path', ''),
        enable_sleep_mode=vllm_config.get('enable_sleep_mode', True),
        tensor_parallel_size=vllm_config.get('inference_tp_size', 2),
        distributed_executor_backend="external_launcher",
        dtype=torch.bfloat16,
        enforce_eager=True,
        gpu_memory_utilization=0.95,
        disable_custom_all_reduce=True,
        skip_tokenizer_init=False,
        max_model_len=2048,
        disable_log_stats=True,
        max_num_batched_tokens=4096,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=19,
    )

    # Resume from checkpoint if specified
    begin_iter = 0
    if cfg.load_path is not None:
        log.info(f"Resuming from checkpoint {cfg.load_path}")
        load_checkpoint(cfg.load_path)
        # Extract iteration from checkpoint name
        ckpt_name = os.path.basename(args.ckpt_path)
        if "ckpt_" in ckpt_name and ".pt" in ckpt_name:
            begin_iter = int(ckpt_name.split("ckpt_")[1].split(".pt")[0])

        # Load weights into vLLM
        load_model_into_vllm(policy_model, inference_engine)

    # Main training loop
    for iteration in trange(begin_iter, cfg.ppo_cfg.num_iterations):
        log.info(f"Iteration {iteration}/{cfg.ppo_cfg.num_iterations}")

        metrics = {}

        # Evaluation
        eval_stats = None
        if iteration % 25 == 0:
            log.info("Evaluating on eval set...")

            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eos_token=EOS_TOKEN,
                eval_sampling_params=SamplingParams(
                    temperature=0.3,
                    max_tokens=1024,
                    n=1,
                    detokenize=False,
                    stop_token_ids=[EOS_TOKEN_ID],
                ),
                reward_func=lambda completion, sample: compute_reward(completion, sample, EOS_TOKEN),
            )
            eval_episode_table = dump_episodes(
                episodes=eval_episodes,
                episodes_stats=eval_stats,
                tokenizer=tokenizer,
                iteration=iteration,
                is_eval=True,
            )
            logger.log({"eval/episodes": eval_episode_table, "iteration": iteration})

        # Sample training batch
        num_samples = cfg.ppo_cfg.episodes_per_iteration // cfg.ppo_cfg.generations_per_sample
        indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
        samples = train_dataset.select(indices)

        gen_time = time.time()

        # Sample responses
        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=cfg.ppo_cfg.generations_per_sample,
                temperature=cfg.ppo_cfg.temperature,
                top_p=cfg.ppo_cfg.top_p,
                top_k=cfg.ppo_cfg.top_k,
                max_tokens=cfg.ppo_cfg.max_response_tokens,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
            ),
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
        inference_engine.sleep(1)

        log.info(f"Generated {len(all_generations)} responses")
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        log.info(f"Time taken to generate {len(all_generations)} responses: {time.time() - gen_time} seconds")

        # Process responses and calculate rewards
        episodes, episodes_stats = create_training_episodes(
            samples,
            all_generations,
            all_finish_reasons,
            tokenizer,
            EOS_TOKEN_ID,
            EOS_TOKEN,
            cfg.ppo_cfg.generations_per_sample,
        )
        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            tokenizer=tokenizer,
            iteration=iteration,
        )

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device=device,
        )

        # Calculate losses and update model
        policy_model.train()
        reference_model.to(device)
        reference_model.eval()

        total_response_len = (model_inputs["labels"] != -100).sum().item()

        train_time = time.time()

        for i in trange(
            0,
            cfg.ppo_cfg.episodes_per_iteration,
            per_device_batch_size,
            desc="Gradient Accumulation",
        ):
            batch = {k: v[i : i + per_device_batch_size] for k, v in model_inputs.items()}

            # Compute policy gradient loss
            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                total_response_len=total_response_len,
                temperature=cfg.ppo_cfg.temperature,
                kl_coefficient=cfg.ppo_cfg.kl_coeff,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())

            # Get gradient norm if available
            if hasattr(policy_model, "get_global_grad_norm"):
                grad_norm = policy_model.get_global_grad_norm()
                if grad_norm is not None:
                    grad_norm = grad_norm.item()
                metrics.setdefault("grad_norm", []).append(grad_norm)

            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v if isinstance(v, (int, float)) else v.item())

            # Backward pass and optimization
            loss.backward()

            # Only free memory if we're at gradient accumulation boundary
            if i % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                reference_model.to('cpu')

            # Free memory
            del loss, loss_metrics

        log.info(f"Time taken to train: {time.time() - train_time} seconds")

        #########################################################
        # Update inference engine weights
        #########################################################

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        #########################################################
        # Log metrics
        #########################################################

        train_metrics = {k: np.mean(v) for k, v in metrics.items() if None not in v}
        if hasattr(policy_model, "get_lr"):
            train_metrics["learning_rate"] = policy_model.get_lr()[0]

        logs = {
            "iteration": iteration,
            **{f"train/{k}": v for k, v in train_metrics.items()},
        }
        if eval_stats is not None:
            logs.update({f"eval/{k}": np.mean(v) for k, v in eval_stats.items()})

        if isinstance(logger, list):
            for log_dest in logger:
                log_dest.log(logs, step=iteration)
        else:
            logger.log(logs)

        selected_keys = [
            "train/kl_penalty",
            "train/rewards",
            "train/reward_metrics/format_reward",
            "train/reward_metrics/equation_reward",
            "eval/rewards",
            "eval/reward_metrics/format_reward",
            "eval/reward_metrics/equation_reward",
        ]
        selected_metrics = {k: logs[k] for k in selected_keys if k in logs}
        log.info(f"KEY METRICS: {selected_metrics}")

if __name__ == "__main__":
    config_path = "yamls/llama-3b-ppo-8gpu.yaml"

    # Load Composer configuration
    from omegaconf import OmegaConf
    with open(config_path) as f:
        config = OmegaConf.load(f)

    main(config)
