import torch
from typing import Dict, Any, Optional

from transformers import AutoModelForCausalLM
from omegaconf.dictconfig import DictConfig
from templates.model import AbstractSentenceAutoEncoder


class SentenceAutoEncoder(AbstractSentenceAutoEncoder):

    def __init__(self, 
                 model_cls: AutoModelForCausalLM,
                 args: DictConfig,
                 ):
        super().__init__()  
        self.args = args
        self.model = model_cls
        self.t = self.model.transformer # the transformer architecture; for gpt this is GPT2Model with 6 * gpt2 block
        self.wte = self.t.wte.weight # token embedding weights (`wte'; torch.Size([vocab_size  n_embs])); for gp2 this is [50257, 768]
        self.n_embs = self.wte.shape[1]
        # create new custom embedding for cmpr and tsk tokens
        self.embds = torch.nn.Embedding( 
            args.training.n_cmps + args.training.n_tsks + int(args.training.sep_cmpr), 
            self.n_embs,
            dtype=torch.floa32 if args.model.dtype == 'float32' else torch.float16,
        ) # shape: (n_cmps + n_tsks + sep_cmpr, n_embs); for gpt2 this is [5, 768]
        if self.wte.get_device() > -1: self.embds.to(self.wte.get_device()) # if device is not cpu, set to gpu index
        # note: skipping args.training.proj_cmpr for now
        self.cmp_ids = [i for i in range(args.training.n_cmps)] # list of compression ids
        self.tsk_ids = [i + args.training.n_cmps for i in range(args.training.n_tsks)] # list of task ids
        self.sep_id = args.training.n_tsks + args.training.n_cmps # seperator id
       
        print(self.get_embeddings())
        self.add_embeddings(1)
    
    @property
    def get_device(self) -> torch.device:
        return self.model.transformer.wte.weight.device
    
    def get_embeddings(self) -> torch.tensor:
        return self.model.transformer.get_input_embeddings()

    def add_embeddings(self,
                       new_embs: int,
                       ) -> None:
        if new_embs <= 0: return
        vocab_size, _ = self.get_embeddings().weight.shape 
        self.model.resize_token_embeddings(vocab_size + new_embs) # increase vocab size

    def compress(self,
                 input_ids: torch.LongTensor,
                 attention_mask: torch.LongTensor,
                 *args,
                 **kwargs) -> torch.tensor:
        
        pass

    def forward(self):
        pass

    def infer(self):
        pass

    def causal_lm(self):
        pass
