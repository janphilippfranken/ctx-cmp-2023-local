import logging

import hydra
import torch
from datasets import DatasetDict, load_dataset
from omegaconf.dictconfig import DictConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from arguments import Arguments, global_setup
# from data import get_data_loaders




@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)
    print(args)
    
    # if args.data.dataset_name == "stas/openwebtext-10k":
    #     lm_datasets = load_dataset(
    #         args.data.dataset_name,
    #         cache_dir=args.model.cache_dir,
    #     )
    # else:
    #     raise NotImplementedError(f"Unknown dataset name {args.data.dataset_name}")
    


if __name__ == "__main__":
    main()
