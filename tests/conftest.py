import pytest
from unittest import mock
import torch
from scuola.config import (Config, ModelConfig, TokenizerConfig, FsdpConfig, TpConfig, VllmConfig, SchedulerConfig, OptimizerConfig, TrainLoaderConfig, DatasetConfig, MlflowConfig, LoggersConfig, PPOConfig)

@pytest.fixture
def mock_tokenizer():
    tokenizer = mock.Mock()
    tokenizer.batch_decode.return_value = [
        "<think>1+2+3=6</think>\n<answer>1+2+3</answer>",
        "invalid"
    ]
    tokenizer.apply_chat_template.return_value = [101, 102]
    tokenizer.decode.return_value = "decoded string"
    tokenizer.eos_token_id = 0
    tokenizer.convert_ids_to_tokens.return_value = "<eos>"
    return tokenizer

@pytest.fixture
def mock_model():
    model = mock.Mock()
    logits = torch.randn((1, 4, 10))
    model.return_value = mock.Mock(logits=logits)
    return model


@pytest.fixture
def mock_llm_engine():
    engine = mock.Mock()
    engine.generate.return_value = [mock.Mock(outputs=[mock.Mock(token_ids=[1, 2], finish_reason="stop")])]
    return engine


@pytest.fixture
def mock_mlflow():
    with mock.patch("scuola.utils.mlflow") as m:
        yield m


@pytest.fixture
def minimal_config():
    return Config(
        model=ModelConfig(),
        tokenizer=TokenizerConfig(),
        fsdp=FsdpConfig(),
        tp=TpConfig(),
        vllm=VllmConfig(),
        scheduler=SchedulerConfig(),
        optimizer=OptimizerConfig(),
        train_loader=TrainLoaderConfig(dataset=DatasetConfig()),
        loggers=LoggersConfig(mlflow=MlflowConfig()),
        ppo=PPOConfig()
    )
