import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Any
from omegaconf import OmegaConf, MISSING
from omegaconf.errors import ConfigAttributeError, ValidationError

MAX_SEQ_LEN = 1024

@dataclass
class ModelConfig:
    name: str = "hf_causal_lm"
    pretrained: bool = True
    pretrained_model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_flash_attention_2: bool = False


@dataclass
class TokenizerConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    kwargs: dict = field(default_factory = lambda : {
            "model_max_length": MAX_SEQ_LEN,
            "trust_remote_code": True
    })


@dataclass
class FsdpConfig:
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: str = "PURE"
    flatten_parameters: bool = True
    forward_prefetch: bool = True
    backward_prefetch: str = "BACKWARD_PRE"
    limit_all_gathers: bool = True
    activation_checkpointing: bool = False
    sync_module_states: bool = True
    forward_sync: bool = True
    use_orig_params: bool = False
    backward_sync: bool = True
    ignored_modules: list[str] = field(default_factory = lambda: [])
    state_dict_type: str = "FULL_STATE_DICT"
    activation_checkpointing_reentrant: bool = False
    cpu_offload: bool = False
    min_params: int = 0
    param_persistence: str = "SELECTIVE"


@dataclass
class TpConfig:
    strategy: str = "unknown"
    tensor_parallel_degree: int = 2


@dataclass
class VllmConfig:
    enabled: bool = True
    strategy: str = "tensor_parallel"
    inference_tp_size: int = 2 # Using tensor parallelism of size 4 on 16 GPUs
    enable_sleep_mode: bool = True # Turn it on for GPU. CPU does not suport sleep mode


@dataclass
class SchedulerConfig:
    name: str = "cosine_with_warmpu"
    t_warmup: str = "50ba"
    alpha_f: float = 0.1
    base_lr: float = 1.0e-6
    min_lr: float = 0.0


@dataclass
class OptimizerConfig:
    name: str = "decoupled_adamw"
    lr: float = 5.0e-6
    betas: list[float] = field(default_factory = lambda: [0.9, 0.999])
    eps: float = 1.0e-8
    weight_decay: float = 0


@dataclass
class DatasetConfig:
    split: str = "train"
    dataset_name: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    max_seq_len: int = MAX_SEQ_LEN
    allow_pad_trimming: bool = False
    decoder_only_format: bool = True
    shuffle: bool = True


@dataclass
class TrainLoaderConfig:
    dataset: DatasetConfig
    name: str = "ppo"
    drop_last: bool = True
    num_workers: int = 8
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    timeout: int = 0


@dataclass
class MlflowConfig:
    tags: dict[str, str] = field(default_factory= lambda: {"group": "test"})
    experiment_name: str = "/Users/xiaohan.zhang@databricks.com/ppo_test"


@dataclass
class LoggersConfig:
    mlflow: MlflowConfig
    log_to_console: bool = True
    progress_bar: bool = False

@dataclass
class PPOConfig:
    kl_coeff: float = 0.1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    max_response_tokens: int = 128
    generations_per_sample: int = 4
    episodes_per_iteration: int = 64
    num_iterations: int = 1000

@dataclass
class Config:
    # Nested configs
    model: ModelConfig
    tokenizer: TokenizerConfig
    fsdp: FsdpConfig
    tp: TpConfig
    vllm: VllmConfig
    scheduler: SchedulerConfig
    optimizer: OptimizerConfig
    train_loader: TrainLoaderConfig
    loggers: LoggersConfig
    ppo: PPOConfig

    # Top level configs
    max_seq_len: int = MAX_SEQ_LEN
    eval_interval: int = 25
    eval_first: bool = True
    global_train_batch_size: int = 32
    seed: int = 17
    device_eval_batch_size: int = 8
    device_train_microbatch_size: Union[int, str] = 8
    precision: str = "amp_bf16"
    dist_timeout: int = 600
    progress_bar: bool = False

    def __repr__(self) -> str:
        """Convert to str via OmegaConf (resolves any remaining references)"""
        return json.dumps(OmegaConf.to_container(OmegaConf.create(asdict(self)), resolve=True), indent=2, default=str)


def load_config(config_path: str) -> Config:
    """
    Load a YAML config file and convert it to a nested Config dataclass using OmegaConf.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config: The fully constructed Config object with all nested dataclasses
    """
    # Load the YAML file using OmegaConf
    config_schema = OmegaConf.structured(Config)
    config_yaml = OmegaConf.load(config_path)

    try:
        # Merge the schema with the loaded config for validation
        merged_config = OmegaConf.merge(config_schema, config_yaml)

        # Convert to nested dataclasses
        return OmegaConf.to_object(merged_config)
    except (ConfigAttributeError, ValidationError) as e:
        raise ValueError(f"Configuration validation error: {e}")


def validate_config(config: Config) -> bool:
    """
    Perform additional sanity checks on the loaded configuration.

    Args:
        config: The Config object to validate

    Returns:
        bool: True if the configuration is valid
    """
    # Add custom validation logic beyond what OmegaConf provides
    # For example:
    if config.max_seq_len <= 0:
        print("Error: max_seq_len must be positive")
        return False

    if config.global_train_batch_size <= 0:
        print("Error: global_train_batch_size must be positive")
        return False

    # Add more validation as needed
    return True


def load_and_validate_config(config_path: str) -> Config:
    """
    Load a YAML config file, convert to dataclasses, and validate.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config: The validated Config object
    """
    config = load_config(config_path)

    if not validate_config(config):
        raise ValueError("Config validation failed")

    return config
