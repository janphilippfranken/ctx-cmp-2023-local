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
        self.t = self.model.transformer
        self.tembs = self.t.get_input_embeddings().weight # torch.Size([vocab_size , dimensionality of the embeddings/hidden states])
        self.hsize = self.tembs.shape[1] # dimensionality of the embeddings/hidden states (eg 768 in gpt2)
        # create new embedding
        self.embds = torch.nn.Embedding(
            self.args.compression.n_cmps + self.args.compression.n_tsks + int(self.args.sep_cmpr), self.hsize
        )
        

    @property
    def device(self) -> torch.device:
        return self.transformer_model.device
    
    def add_attrs(self, new_attrs: Dict[str, Any]) -> None:
        pass

    def get_embeddings(self) -> torch.tensor:
        pass

    def add_embeddings(self,
                       n_embs: int) -> None:
        pass

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
