import pytest
import torch
import torch.nn as nn
import mlflow
from unittest import mock
from datasets import Dataset

from scuola.ppo.ppo import (
    format_reward_func, equation_reward_func,
    compute_reward, create_training_episodes,
    compute_pg_loss, preprocess_example, data_prep
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, input_ids, attention_mask):
        # fake embedding of input_ids shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, 10, requires_grad=True)
        logits = self.linear(hidden)
        return type('Output', (), {'logits': logits})


def test_format_reward_func():
    valid_completion = "<think>solved</think>\n<answer>1+2+3</answer>"
    valid_completion2= "Let me solve this step by step.\n<think>\nResp:     We have the numbers 20, 21, 64, and 4. We need to create an equation that equals 57. Let's start by trying to use each number once. </think>\n<answer>(64 - 20) / (4 + 21)</answer><|im_end|>"
    invalid_completion = "no tags"
    assert format_reward_func(valid_completion, "-1") == 1.0
    assert format_reward_func(valid_completion2, "-1") == 1.0
    assert format_reward_func(invalid_completion, "-1") == 0.0


def test_equation_reward_func():
    correct = "<answer>1+2+3</answer>"
    incorrect = "<answer>1*2</answer>"
    assert equation_reward_func(correct, [1,2,3], 6) == 1.0
    assert equation_reward_func(incorrect, [1,2,3], 6) == 0.0


def test_compute_reward():
    completion = "<think>...</think>\n<answer>1+2+3</answer>"
    reward, details = compute_reward(completion, {"nums": [1,2,3], "target": 6}, eos_token="-1")
    assert reward == 2.0
    assert details == {"format_reward": 1.0, "equation_reward": 1.0}


def test_create_training_episodes(mock_tokenizer):
    samples = Dataset.from_dict({
        "input_ids": [[101, 1]], "nums": [[1,2]], "target": [3]
    })
    episodes, stats = create_training_episodes(
        samples,
        all_generations=[[200, 201], [300]],
        all_finish_reasons=["stop", "length"],
        tokenizer=mock_tokenizer,
        eos_token_id=0,
        eos_token="<eos>",
        generations_per_sample=2
    )
    assert len(episodes["all_query_token_ids"]) == 2
    assert "rewards" in stats


def test_compute_pg_loss():
    model = DummyModel()
    batch = {
        "input_ids": torch.tensor([[1,2,3]]),
        "attention_mask": torch.tensor([[1,1,1]]),
        "labels": torch.tensor([[-100,2,3]]),
        "advantages": torch.tensor([[0.5, 0.5, 0.5]])
    }
    loss, metrics = compute_pg_loss(
        policy_model=model,
        reference_model=model,
        batch=batch,
        total_response_len=2,
        temperature=1.0,
        kl_coefficient=0.1
    )
    print(f"I am here {loss.requires_grad=}")
    assert loss.requires_grad
    assert "policy_loss" in metrics
    del model


