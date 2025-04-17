# utils.py
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from typing import Any, Dict, List, Tuple, Callable

from vllm import LLM, SamplingParams
from datasets import Dataset
import numpy as np

from transformers import AutoTokenizer, PreTrainedModel

###############################################################################
# Prepare Model Inputs
###############################################################################
def prepare_model_inputs(
    query_token_ids: List[List[int]],
    response_token_ids: List[List[int]],
    advantages: List[List[float]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Same logic: pad queries + responses into a single sequence, build masks, build label, etc.
    """
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids))
    pad_token_id = 0
    ignore_index = -100

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    advantages_list = []

    for q_ids, r_ids, adv_list in zip(query_token_ids, response_token_ids, advantages):
        combined_ids = q_ids + r_ids
        seq_len = len(combined_ids)

        padded_input_ids = [pad_token_id]*(max_seq_len - seq_len) + combined_ids
        padded_attention = [0]*(max_seq_len - seq_len) + [1]*seq_len

        # Labels: mask out the query tokens with -100
        padded_labels = [ignore_index]*(max_seq_len - seq_len) + [ignore_index]*len(q_ids) + r_ids

        # Advantages: 0.0 for query tokens, advantage for response tokens, pad the rest
        padded_adv = [0.0]*(max_seq_len - seq_len) + [0.0]*len(q_ids) + adv_list

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention)
        labels_list.append(padded_labels)
        advantages_list.append(padded_adv)

    # Convert
    input_ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    attention_mask_t = torch.tensor(attention_mask_list, dtype=torch.long, device=device)
    labels_t = torch.tensor(labels_list, dtype=torch.long, device=device)
    # BFloat16 or float16 or float?
    advantages_t = torch.tensor(advantages_list, dtype=torch.bfloat16, device=device)

    print(f"I am here 20: {advantages_t.dtype=}")

    return {
        "input_ids": input_ids_t,
        "attention_mask": attention_mask_t,
        "labels": labels_t,
        "advantages": advantages_t,
    }

###############################################################################
# Compute Token Log Probs
###############################################################################
def compute_token_log_probs(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    1. Forward pass for causal LM
    2. Shifted logits (one step)
    3. Gather log probs at the actual next token
    4. Return logprobs
    """
    # forward pass
    # We expect a standard HF Causal LM that returns `logits`
    out = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    logits = out.logits
    # Temperature scale
    logits = logits.float() / temperature

    # shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous()

    # mask
    label_mask = (shift_labels != -100).float()
    shift_labels[shift_labels == -100] = 0  # just to avoid gather on -100

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(2)).squeeze(2)
    log_probs = log_probs * label_mask
    return log_probs


###############################################################################
# Evaluate on Test Set
###############################################################################
def evaluate_on_test_set(
    inference_engine: LLM,
    test_dataset: Dataset,
    tokenizer: AutoTokenizer,
    eos_token: str,
    eval_sampling_params: SamplingParams,
    reward_func: Callable[[str, Dict[str, Any]], Tuple[float, Dict[str, float]]],
    local_rank: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Use vLLM to generate from test set prompts, compute reward.
    """
    print(f"I am here 1")
    # Generate
    generations = inference_engine.generate(
        prompt_token_ids=test_dataset["input_ids"],
        sampling_params=eval_sampling_params,
    )

    print(f"I am here 2")
    metrics = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    all_query_token_ids = []
    all_responses_token_ids = []

    print(f"I am here 3")
    for i, sample in enumerate(test_dataset):
        q_ids = sample["input_ids"]
        # vLLM returns [Out...], so for each i, we have one set of outputs
        response_token_ids = generations[i].outputs[0].token_ids
        finish_reason = generations[i].outputs[0].finish_reason

        response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        rew, rew_components = reward_func(response, sample)

        all_query_token_ids.append(q_ids)
        all_responses_token_ids.append(response_token_ids)

        metrics["response_lengths"].append(len(response_token_ids))
        metrics["rewards"].append(rew)
        metrics["non_stop_rate"].append(finish_reason != "stop")
        for k, v in rew_components.items():
            metrics.setdefault(f"reward_metrics/{k}", []).append(v)

    print(f"I am here 4")
    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
    }
    return episodes, metrics


###############################################################################
# Dump Episodes (example table)
###############################################################################
def dump_episodes(
    logger,  # ignoring logger here for minimal code
    episodes: Dict[str, Any],
    episodes_stats: Dict[str, Any],
    tokenizer: AutoTokenizer,
    iteration: int,
    is_eval: bool = False,
) -> List[List[Any]]:
    """
    Print a few examples for debug.
    If you want to store to MLflow or something, do it here.
    """
    q_ids = episodes["all_query_token_ids"]
    r_ids = episodes["all_response_token_ids"]

    rewards = episodes_stats.get("rewards", [])
    lengths = episodes_stats.get("response_lengths", [])

    q_texts = tokenizer.batch_decode(q_ids, skip_special_tokens=False)
    r_texts = tokenizer.batch_decode(r_ids, skip_special_tokens=False)

    # Just print first 2 examples
    if len(q_texts) >= 2 and not is_eval:
        print(f"##### Iteration {iteration} Example 1")
        print(f"Query:   {q_texts[0]}")
        print(f"Resp:    {r_texts[0]}")
        print(f"Reward:  {rewards[0] if len(rewards)>0 else 'n/a'}")
        print("")

        print(f"##### Iteration {iteration} Example 2")
        print(f"Query:   {q_texts[1]}")
        print(f"Resp:    {r_texts[1]}")
        print(f"Reward:  {rewards[1] if len(rewards)>1 else 'n/a'}")
        print("")

    table = []
    for i in range(len(q_texts)):
        rew = rewards[i] if i < len(rewards) else None
        length = lengths[i] if i < len(lengths) else None
        table.append([q_texts[i], r_texts[i], rew, length])
    return table


###############################################################################
# Load Weights into vLLM
###############################################################################
def load_model_into_vllm(model: FSDP, llm: LLM) -> None:
    """
    Copy the (fully sharded) model state into the vLLM engine.
    """
    world_size = dist.get_world_size()

    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    # FSDP: collect a FULL state_dict
    state_dict = model.state_dict()  # automatically merges shards

    # vLLM has a .load_weights(...) that expects an iterable of (name, param).
    # If world_size > 1, each param might be sharded, but FSDP's state_dict merges them for you.
    params = ((name, param) for name, param in state_dict.items())

    loaded_params = vllm_model.load_weights(params)
    if dist.get_rank() == 0:
        print(f"vLLM loaded {len(loaded_params)} param partitions.")


###############################################################################
# Optional: MLflow
###############################################################################
def init_mlflow(*args, **kwargs):
    """
    Example stub if you want to integrate MLflow logging.
    Called once at the start by rank0, etc.
    """
    pass

