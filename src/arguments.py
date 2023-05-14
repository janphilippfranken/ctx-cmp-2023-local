"""
Based on https://github.com/jayelm/gisting/blob/main/src/arguments.py.
"""
import logging
import os.path as osp
import socket
from dataclasses import dataclass, field
from typing import Optional, List

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Model Arguments.
    """
    model_name_or_path: str = field(
        default="bigscience/bloomz-560m", 
        metadata={
            "help": "The model checkpoint for weights initialization."
        }
    )
    cache_dir: Optional[str] = field(
        default="/data/jphilipp/research-projects/ctx-cmp-2023/.cache/huggingface",
        metadata={
            "help": (
                "Where to store pre-trained models and datasets."
            )
        },
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Training Arguments with type fixes. 
    """
    pass
    def __post_init__(self):
        # Don't run post-init until ready to convert to TrainingArgs
        if self._run_post_init:
            super().__post_init__()


@dataclass
class DataArguments:
    """
    Data Arguments.
    """
    dataset_name: Optional[str] = field(
        default="stas/openwebtext-10k",
        metadata={"help": "The name of the dataset for datasets.load_dataset(dataset_name)."},
    )
    load_split: Optional[str] = field(
        default="train",
        metadata={"help": "The split of the dataset loaded using datasets.load_dataset()."},
    )
    num_proc: Optional[int] = field(
        default=4,
        metadata={
            "help": (
                "Multiprocessing significantly speeds up processing by parallelizing processes on the CPU."
                "Sets the num_proc parameter in map() to set the number of processes to use:"
            )
        },
    )
    batched: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                 "Operate on batches by setting batched=True. The default batch size is 1000, but you can adjust it with the batch_size parameter." 
            )
        },
    )
    batch_size: Optional[int] = field(
        default=1000,
        metadata={
            "help": (
                 "Size of each batch to be processed by map()." 
            )
        },
    )
    remove_columns: Optional[List[str]] = field(
        default_factory=lambda: ["text"],
        metadata={
            "help": (
            "Columns to remove from the dataset after tokenization."
            )
        },
    )


@dataclass
class Arguments:
    model: ModelArguments = ModelArguments()
    # training: CustomTrainingArguments = CustomTrainingArguments()
    data: DataArguments = DataArguments() 


cs = ConfigStore.instance()
cs.store(name="base_config", node=Arguments)


def global_setup(args: DictConfig) -> Arguments:
    """Global setup of arguments."""
    # Print the hostname
    hostname = socket.gethostname()
    logger.info(f"Running on {hostname}")

    args = OmegaConf.to_object(args)

    return args
