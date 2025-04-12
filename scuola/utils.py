import json
import socket
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer, PreTrainedModel
from composer import ComposerModel
from composer.loggers.mlflow_logger import MLFlowLogger
from composer.utils import dist
from vllm import LLM, SamplingParams

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
DEFAULT_PROMPT_TEMPLATE = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."


def create_prompt(
    numbers: List[int],
    target: int,
    tokenizer: AutoTokenizer,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    prefix = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": prompt_template.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    return tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)


def prepare_model_inputs(
    query_token_ids: List[List[int]],
    response_token_ids: List[List[int]],
    advantages: List[List[float]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Prepare padded model inputs with attention masks, labels, and advantages.
    Args:
        query_token_ids: List of query token ids
        response_token_ids: List of response token ids
        advantages: List of lists of advantage values, matching response_token_ids structure
        device: Device to move the tensors to
    Returns:
        Dict with input_ids, attention_mask, labels, and advantages

    Example:
        >>> query_token_ids = [[1, 2, 3], [4, 5]]
        >>> response_token_ids = [[6, 7], [8]]
        >>> advantages = [[0.5, 0.8], [0.3]]
        >>> outputs = prepare_model_inputs(query_token_ids, response_token_ids, advantages, "cuda")
        >>> outputs
        {
            'input_ids': tensor([
                [1, 2, 3, 6, 7],
                [4, 5, 8, 0, 0]
            ]),
            'attention_mask': tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0]
            ]),
            'labels': tensor([
                [-100, -100, -100, 6, 7],
                [-100, -100, 8, -100, -100]
            ]),
            'advantages': tensor([
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.9, 0.0]
            ])
        }
    """
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids))
    inputs = {"input_ids": [], "attention_mask": [], "labels": [], "advantages": []}

    pad_token_id = 0  # Doesn't matter, will be masked
    ignore_index = -100

    for query, response, advantage in zip(query_token_ids, response_token_ids, advantages):
        combined_ids = query + response
        seq_len = len(combined_ids)

        # Create padded sequences
        input_ids = combined_ids + [pad_token_id] * (max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        labels = [ignore_index] * len(query) + response + [ignore_index] * (max_seq_len - seq_len)
        advantages_seq = [0.0] * len(query) + advantage + [0.0] * (max_seq_len - seq_len)

        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(labels) == max_seq_len
        assert len(advantages_seq) == max_seq_len

        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append(attention_mask)
        inputs["labels"].append(labels)
        inputs["advantages"].append(advantages_seq)

    # Convert to tensors
    return {
        k: torch.tensor(v, dtype=torch.long if k != "advantages" else torch.float, device=device)
        for k, v in inputs.items()
    }


def compute_token_log_probs(
    model: Union[ComposerModel],
    inputs: Dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """
    Compute log probabilities for each token in the sequence, masked for valid labels only.

    This function:
    1. Runs the model forward pass
    2. Applies temperature scaling to logits
    3. Shifts the sequences for causal language modeling
    4. Computes log probabilities for the actual tokens that appeared in the sequence
    5. Masks the log probabilities to only include valid labels (non -100 positions)

    Args:
        model: The language model (ComposerModel)
        inputs: Dictionary containing:
            - input_ids: Tensor of token ids [batch_size, seq_len]
            - attention_mask: Tensor of attention mask [batch_size, seq_len]
            - labels: Tensor of target labels [batch_size, seq_len] with -100 for ignored positions
        temperature: Temperature for scaling the logits before softmax

    Returns:
        torch.Tensor: Log probabilities tensor of shape [batch_size, seq_len-1], where:
            - Each value is the log probability of the actual token that appeared
            - Values are masked to 0.0 for positions where labels were -100
            - The sequence length is reduced by 1 due to the causal shift

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = {
        ...     "input_ids": torch.tensor([[1, 2, 3]]),
        ...     "attention_mask": torch.tensor([[1, 1, 1]]),
        ...     "labels": torch.tensor([[-100, 2, 3]])
        ... }
        >>> log_probs = compute_token_log_probs(model, inputs, temperature=1.0)
        >>> log_probs.shape
        torch.Size([1, 2])  # batch_size=1, seq_len-1=2
        >>> # First position is 0 (masked), second position has actual log prob
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )

    logits = outputs.logits.float() / temperature  # Shape: [batch_size, seq_len, vocab_size]
    shift_logits = logits[..., :-1, :].contiguous()  # Shape: [batch_size, seq_len-1, vocab_size]
    shift_labels = inputs["labels"][..., 1:].contiguous()  # Shape: [batch_size, seq_len-1]

    # Create mask for valid labels
    label_mask = (shift_labels != -100).float()  # Shape: [batch_size, seq_len-1]
    shift_labels[shift_labels == -100] = 0  # Shape: [batch_size, seq_len-1]

    # Calculate log probabilities
    log_probs = torch.log_softmax(shift_logits, dim=-1)  # Shape: [batch_size, seq_len-1, vocab_size]
    log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2))  # Shape: [batch_size, seq_len-1, 1]
    log_probs = log_probs.squeeze(2)  # Shape: [batch_size, seq_len-1]
    log_probs = log_probs * label_mask  # Shape: [batch_size, seq_len-1]

    return log_probs


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def evaluate_on_test_set(
    inference_engine: LLM,
    test_dataset: Dataset,
    tokenizer: AutoTokenizer,
    eos_token: str,
    eval_sampling_params: SamplingParams,
    reward_func: Callable[[str, Dict[str, Any]], Tuple[float, Dict[str, float]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate the model on a test dataset by generating responses and computing rewards.

    Args:
        inference_engine: The sglang Engine instance used for text generation
        test_dataset: Dataset containing test samples
        tokenizer: Tokenizer for decoding generated token IDs
        eos_token: End of sequence token string
        eval_sampling_params: Dictionary of parameters for controlling the generation process
        reward_func: Function that computes rewards for generated responses. Takes a response
            string and sample dict as input, returns a tuple of (overall_reward, reward_components)

    Returns:
        Dictionary containing evaluation statistics:
            - response_lengths: List of token counts for each generated response
            - rewards: List of overall reward values for each response
            - non_stop_rate: List of booleans indicating if generation ended for non-stop reason
            - reward_metrics/*: Lists of individual reward component values, prefixed with
              "reward_metrics/"
        episodes: Dictionary containing:
            - all_query_token_ids: List of query token IDs for each episode
            - all_response_token_ids: List of response token IDs for each episode

    Example:
        >>> episodes, episodes_stats = evaluate_on_test_set(
        ...     inference_engine=engine,
        ...     test_dataset=dataset,
        ...     tokenizer=tokenizer,
        ...     eos_token="</s>",
        ...     eval_sampling_params={"temperature": 0.7, "max_tokens": 100},
        ...     reward_func=compute_rewards
        ... )
        >>> print(f"Average reward: {episodes_stats['rewards']:.3f}")
    """
    generations = inference_engine.generate(
        prompt_token_ids=test_dataset["input_ids"], sampling_params=eval_sampling_params
    )

    metrics = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    all_query_token_ids = []
    all_responses_token_ids = []

    for i, sample in enumerate(test_dataset):
        query_token_ids = sample["input_ids"]
        response_token_ids = generations[i].outputs[0].token_ids
        finish_reason = generations[i].outputs[0].finish_reason

        response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        reward, reward_components = reward_func(response, sample)

        all_query_token_ids.append(query_token_ids)
        all_responses_token_ids.append(response_token_ids)

        metrics["rewards"].append(reward)
        metrics["non_stop_rate"].append(finish_reason != "stop")
        metrics["response_lengths"].append(len(response_token_ids))
        for k, v in reward_components.items():
            metrics.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
    }

    return episodes, metrics



def dump_episodes(
    logger: MLFlowLogger,
    episodes: Dict[str, Any],
    episodes_stats: Dict[str, Any],
    tokenizer: AutoTokenizer,
    iteration: int,
    is_eval: bool = False,
) -> list[list[Any]]:
    query_token_ids = episodes["all_query_token_ids"]
    response_token_ids = episodes["all_response_token_ids"]
    rewards = episodes_stats["rewards"]
    response_lengths = episodes_stats["response_lengths"]

    query_texts = tokenizer.batch_decode(
        query_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    response_texts = tokenizer.batch_decode(
        response_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if not is_eval:
        print(f"########## Example 1 (Reward: {rewards[0]}, Response Length: {response_lengths[0]})")
        print(f"#### Query:\n`{query_texts[0]}`")
        print(f"#### Response:\n`{response_texts[0]}`\n\n")

        print(f"########## Example 2 (Reward: {rewards[1]}, Response Length: {response_lengths[1]})")
        print(f"#### Query:\n`{query_texts[1]}`")
        print(f"#### Response:\n`{response_texts[1]}`\n\n")

    table = []
    for i in range(len(query_texts)):
        table.append([query_texts[i], response_texts[i], rewards[i], response_lengths[i]])

    logger.log_table(columns=["query", "response", "reward", "response_length"], rows=table)

    return table



def load_model_into_vllm(model: ComposerModel, llm: LLM) -> None:
    """
    Load weights from a ComposerModel into a vLLM inference engine.

    This function transfers the weights from a training model to a vLLM inference engine,
    allowing for efficient inference using the updated model weights.

    Args:
        model (ComposerModel): The source model to copy weights from.
        vllm (LLM): The target vLLM inference engine to load the weights into.
            Must be already initialized and ready to accept new weights.

    Returns:
        None
    """
    state_dict = model.module.state_dict() if isinstance(model, ComposerModel) else model.state_dict()
    FSDP.set_state_dict_type(module,
                             state_dict_type=StateDictType.FULL_STATE_DICT,
                             state_dict_config=FullStateDictConfig())
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())



class PPOMLFlowLogger(MLFlowLogger):
    def init(self) -> None:

        if self.run_name is None:
            self.run_name = state.run_name

        self._global_exception_occurred = 0

        # Store the Composer run name in the MLFlow run tags so it can be retrieved for autoresume
        self.tags['run_name'] = os.environ.get('RUN_NAME', state.run_name)

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            self.run_name += f'-rank{dist.get_global_rank()}'

        # Register the global exception handler so that uncaught exception is tracked.
        original_excepthook = sys.excepthook
        sys.excepthook = lambda exc_type, exc_value, exc_traceback: self._global_exception_handler(
            original_excepthook,
            exc_type,
            exc_value,
            exc_traceback,
        )
        # Start run
        if self._enabled:
            self._start_mlflow_run(state)

        # If rank zero only, broadcast the MLFlow experiment and run IDs to other ranks, so the MLFlow run info is
        # available to other ranks during runtime.
        if self._rank_zero_only:
            mlflow_ids_list = [self._experiment_id, self._run_id]
            dist.broadcast_object_list(mlflow_ids_list, src=0)
            self._experiment_id, self._run_id = mlflow_ids_list
