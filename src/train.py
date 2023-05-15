# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training script, adapted from huggingface's run_clm.py example and jesse's train.py
"""
import logging
import os

import hydra
import torch
from datasets import DatasetDict, load_dataset
from omegaconf.dictconfig import DictConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from arguments import Arguments, global_setup
from data import CompressionTokenizer

# Will error if the minimal version of Transformers is not installed. 
check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)

    # 1 Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.training.output_dir)
        and args.training.do_train
        and not args.training.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.training.output_dir)
        if last_checkpoint is None and len(os.listdir(args.training.output_dir)) > 0:
            existing_files = os.listdir(args.training.output_dir)
            logger.warning(
                (
                    "Output directory (%s) already exists and "
                    "is not empty. Existing files: %s. "
                    "Training anyways as these may just be output files."
                ),
                args.training.output_dir,
                str(existing_files),
            )
        elif (
            last_checkpoint is not None and args.training.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from "
                "scratch."
            )

    # 2 Set seed before initializing model.
    set_seed(args.training.seed)

    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
    }

    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name, **tokenizer_kwargs
        )
    elif args.model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.model_name_or_path, **tokenizer_kwargs
        )

    # 3 Load datasets.
    cmpr_tokenizer = CompressionTokenizer(tokenizer, args)
    dataset, valset, data_loader, val_loader = cmpr_tokenizer.get_data_loaders()
    print(dataset)
    print(valset)
    print(data_loader)
    print(val_loader)
   
    


# model_checkpoint = "bigscience/bloomz-560m"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    


if __name__ == "__main__":
    main()
