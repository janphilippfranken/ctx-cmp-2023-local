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
Training script, adapted from huggingface's run_clm.py example and from https://github.com/jayelm/gisting
"""
import logging
import os


import hydra
import torch
import math
from omegaconf.dictconfig import DictConfig
from transformers import (
    AutoTokenizer,
    AutoConfig, # need to add later
    AutoModelForCausalLM,
    Trainer, 
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from arguments import Arguments, global_setup
from data import CompressionTokenizer
from model import SentenceAutoEncoder

# Will error if the minimal version of Transformers is not installed. 
check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


RMB = "|<RMB>|" # Extra characters are to ensure uniqueness
CMP = "|<CMP{}>|"
SOS = "|<SOS>|"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)

    # Detecting last checkpoint (currently not utilised).
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

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # 1 Tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
        "truncation_side": args.model.truncation_side,
    }
    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name, **tokenizer_kwargs
        )
    elif args.model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.model_name_or_path, **tokenizer_kwargs
        )
    # check for special tokens (need to chat how this relates to the different models' tokenizers and make it more clear)
    num_added = 0
    if tokenizer.pad_token is None:
        print(f"No Pad Token in {args.model.model_name_or_path}")
        print(f"EOS: {tokenizer.eos_token}")
        print(f"CLS: {tokenizer.cls_token}")
        print(f"SEP: {tokenizer.sep_token}")
        if tokenizer.eos_token is not None: # this is unclear 
            tokenizer.add_special_tokens(
                {"pad_token": tokenizer.eos_token}
            )
        else:
            num_added += tokenizer.add_special_tokens(
                {"pad_token": "|<PAD>|"}
            )
         #num_added += tokenizer.add_special_tokens({ ????
        #    "pad_token": "|<PAD>|",
        #    "eos_token": "|<EOS>|",
        #})

    args.model.pad_token = tokenizer.pad_token

    # 2 Datasets
    cmpr_tokenizer = CompressionTokenizer(tokenizer, args)
    tokenized_dataset, data_loader, val_loader = cmpr_tokenizer.get_data_loaders()

    # 3 Model 
    is_bloom = "bloomz" in args.model.model_name_or_path.lower().replace('/', '-').split('-')
    is_gpt2 = "distilgpt2" in args.model.model_name_or_path.lower().replace('/', '-').split('-') # for debugging on cpu 

    if is_bloom:
        model_cls = AutoModelForCausalLM.from_pretrained(args.model.model_name_or_path,
                                                         torch_dtype=torch.float32 if args.model.dtype == 'float32' else torch.float16)                                                
    elif is_gpt2:
        model_cls = AutoModelForCausalLM.from_pretrained(args.model.model_name_or_path,
                                                         torch_dtype=torch.float32 if args.model.dtype == 'float32' else torch.float16)
    else:
        raise ValueError(f"Model type {args.model.model_name_or_path} not supported")
    
    custom_lm = SentenceAutoEncoder(model_cls, args)
    if num_added > 0: custom_lm.add_embeddings(num_added) # resize vocab size if new special tokens have been added

    # 4 Training
    trainer = Trainer( # TODO: understand what we need to customise https://huggingface.co/blog/pytorch-ddp-accelerate-transformers
        model=custom_lm.model, # this is prob wrong as its just the original tranformer? TODO: check. NOTE: it has the new (embs) layers added, so def architecture has changed now.
        args=args.training,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
    )   

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == "__main__":
    main()
