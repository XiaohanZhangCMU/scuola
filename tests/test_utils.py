import pytest
import torch
from unittest import mock
from datasets import Dataset

from scuola.config import MlflowConfig
from scuola.utils import (
    prepare_model_inputs, compute_token_log_probs, evaluate_on_test_set,
    dump_episodes, load_model_into_vllm,
    mlflow_initialize, mlflow_log_params, mlflow_log_metrics, mlflow_end_run
)

def test_prepare_model_inputs():
    inputs = prepare_model_inputs(
        query_token_ids=[[1, 2]],
        response_token_ids=[[3, 4]],
        advantages=[[0.5, 0.5]],
        device=torch.device("cpu")
    )
    assert "input_ids" in inputs
    assert inputs["advantages"].shape == (1, 4)


def test_compute_token_log_probs(mock_model):
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[-100, 2, 3]])
    }
    log_probs = compute_token_log_probs(mock_model, inputs, temperature=1.0)
    assert log_probs.shape == (1, 2)


def test_evaluate_on_test_set(mock_llm_engine, mock_tokenizer):
    test_ds = Dataset.from_dict({"input_ids": [[1, 2]], "nums": [[1,2,3]], "target": [6]})
    episodes, metrics = evaluate_on_test_set(
        inference_engine=mock_llm_engine,
        test_dataset=test_ds,
        tokenizer=mock_tokenizer,
        eos_token="<eos>",
        eval_sampling_params=None,
        reward_func=lambda x, y: (1.0, {}),
        local_rank=0
    )
    assert len(episodes["all_query_token_ids"]) == 1
    assert metrics["rewards"] == [1.0]


def test_dump_episodes(mock_tokenizer):
    episodes = {
        "all_query_token_ids": [[1,2]],
        "all_response_token_ids": [[3,4]]
    }
    stats = {"rewards": [1.0], "response_lengths": [2]}
    table = dump_episodes(None, episodes, stats, mock_tokenizer, iteration=1)
    print(f"{table=}")
    assert len(table) == 2
    assert table[0][2] == 1.0
    assert table[0][3] == 2



def test_mlflow_initialize(mock_mlflow):
    cfg = MlflowConfig(tags={"group": "test"}, experiment_name="test_exp", log_system_metrics=True)

    with mock.patch.dict("os.environ", {"DATABRICKS_TOKEN": "fake", "DATABRICKS_HOST": "fake"}):
        with mock.patch("torch.distributed.is_initialized", return_value=True), \
             mock.patch("torch.distributed.get_rank", return_value=0), \
             mock.patch("mlflow.set_tracking_uri"), \
             mock.patch("mlflow.set_experiment"), \
             mock.patch("mlflow.set_system_metrics_samples_before_logging"), \
             mock.patch("mlflow.set_system_metrics_sampling_interval"), \
             mock.patch("mlflow.start_run"), \
             mock.patch("scuola.utils.MlflowClient") as mock_client_class:

            mock_client = mock.Mock()
            mock_exp = mock.Mock()
            mock_exp.experiment_id = "exp-123"
            mock_client.get_experiment_by_name.return_value = mock_exp
            mock_client.create_run.return_value = mock.Mock(info=mock.Mock(run_id="run-456"))
            mock_client_class.return_value = mock_client

            mlflow_initialize(cfg)

            mock_client_class.assert_called_once()
            mock_client.create_run.assert_called_once()


def test_mlflow_log_params(mock_mlflow, minimal_config):
    mlflow_log_params(minimal_config)
    mock_mlflow.log_params.assert_called()


def test_mlflow_log_metrics(mock_mlflow):
    with mock.patch("scuola.utils.log_metrics") as mock_log_metrics:
        mlflow_log_metrics({"metric": 1.0}, step=1)
        mock_log_metrics.assert_called_once_with(metrics={"metric": 1.0}, step=1, synchronous=True)


def test_mlflow_end_run(mock_mlflow):
    mlflow_end_run()
    mock_mlflow.end_run.assert_called_once()


