import torch
import torch.nn.functional as F
from typing import (
    Dict, 
    Any, 
    Optional,
)
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

        self.wte = self.t.get_input_embeddings().weight  # token embedding weights (`wte'; torch.Size([vocab_size, n_embs]))
        self.n_embs = self.wte.shape[1]
        
        # custom embedding for cmpr and tsk tokens
        embedding_size = args.training.n_cmps + args.training.n_tsks + int(args.training.sep_cmpr)
        embedding_dtype = torch.float32 if args.model.dtype == 'float32' else torch.float16
        self.embs = torch.nn.Embedding(embedding_size, self.n_embs, dtype=embedding_dtype)

        # set device 
        if self.wte.get_device() > -1: 
            self.embs.to(self.wte.get_device())  

        # compression ids, task ids and the separator id
        self.cmp_ids = list(range(args.training.n_cmps))
        self.tsk_ids = list(range(args.training.n_cmps, args.training.n_cmps + args.training.n_tsks))
        self.sep_id = args.training.n_tsks + args.training.n_cmps  

        data = {'input_ids': [2953, 262, 3726, 286, 262, 10037, 444, 8200, 5701, 1097, 16009, 2275, 11999, 373, 19233, 656, 262, 23327, 13735, 543], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'output_ids': [2950, 2406, 739, 262, 6355, 4811, 4634, 34756, 416, 23327, 447, 247, 82, 1353, 2974, 286, 4542, 13, 2275, 11999, 1719, 550, 257, 2383, 2106, 287, 5584, 6332, 340, 373, 3066, 326, 21534, 544, 290, 2275, 11999, 561, 12082, 3386, 290, 2962, 319, 36467, 13, 632, 373, 257, 640, 618, 23327, 373, 14771, 355, 881, 1637, 656, 36467, 355, 584, 2706, 547, 14771, 287, 19639, 352, 13, 5856, 777, 10861, 812, 2275, 11999, 11949, 43737, 48590, 28607, 445, 72, 2727, 257, 2168, 286, 7903, 6300, 286, 3227, 23327, 5006, 351, 12476, 319, 262, 19755, 290, 262, 23134, 11, 290, 2275], 'output_attn_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': [2953, 262, 3726, 286, 262, 10037, 444, 8200, 5701, 1097, 16009, 2275, 11999, 373, 19233, 656, 262, 23327, 13735, 543]}
        input_ids = torch.LongTensor([data['input_ids'], data['input_ids'], data['input_ids'], data['input_ids']])
        attention_masks = torch.LongTensor([data['attention_mask'], data['attention_mask'], data['attention_mask'], data['attention_mask']])
        output_ids = torch.LongTensor([data['output_ids'], data['output_ids'], data['output_ids'], data['output_ids']])
        output_attn_mask = torch.LongTensor([data['output_attn_mask'], data['output_attn_mask'], data['output_attn_mask'], data['output_attn_mask']])

        self.forward({'input_ids': input_ids, 'attention_mask': attention_masks, 'output_ids': output_ids, 'output_attn_mask': output_attn_mask})
    
    @property
    def get_device(self) -> torch.device:
        return self.model.transformer.get_input_embeddings().weight.device
    
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
        """
        Compresses the input ids to a single vector. See BaseClass for details (Ignoring train_embs and sep_cmpr for now).
        """
        model_embs = self.get_embeddings()  
        inpt_embs = model_embs(input_ids)  

        cmp_embs = self.embs.weight[self.cmp_ids[0]:self.cmp_ids[-1]+1]  
        cmp_embs = cmp_embs[None].repeat(len(inpt_embs), 1, 1) 

        inpt_embs = torch.cat([inpt_embs, cmp_embs], dim=1)
        
        # Pad attention_mask to match new input_emb dimension
        pad_dim = (0, self.args.training.n_cmps + int(self.args.training.sep_cmpr))
        attention_mask = F.pad(attention_mask, pad_dim) 

        fx = self.model.transformer( # see BaseModelOutputWithPastAndCrossAttentions (https://huggingface.co/docs/transformers/main_classes/output)
            inputs_embeds=inpt_embs,
            attention_mask=attention_mask,
            output_hidden_states=True, 
        ) 
        # Select the representational layer of choice, defaulting to the last layer
        n_layers = len(fx["hidden_states"])
        layer_dict = {'half': n_layers//2, None: -1}
        layer = self.args.model.cmp_layer
        chosen_layer = layer_dict.get(layer, layer)
        cmpr = fx['hidden_states'][chosen_layer][:,-self.args.training.n_cmps:] # shape: (batch_size, n_cmps, hidden_size) where n_cmps are the final n_cmps hidden states of the chosen layer

        return cmpr

    def forward(self,
                data: Dict[str, torch.LongTensor],
                ) -> torch.tensor:
        """
        Runs the forward pass. See BaseClass for details.
        """
        cmpr = self.compress(**data)
        if not self.args.training.tforce:
            raise NotImplementedError(f"Not tfoce has to be implemented")
        model_embs = self.get_embeddings()  
        out_embs =  model_embs(data["output_ids"]) # shape (batch_size, seq_len, hidden_size)
        sos = self.embs.weight[self.tsk_ids[0]][None,None] # sos token
        local_rank = self.args.training.local_rank if self.args.training.local_rank != -1 else "cpu" # uncomment / remove for gpu
        out_embs = torch.cat( 
            [
                cmpr.to(local_rank),
                sos.repeat(len(cmpr),1,1),
                out_embs.to(local_rank),
            ],
            dim=1,
        ) # shape (batch_size, n_cmps + sos + seq_len, hidden_size)
        npad = out_embs.shape[1] - data["output_attn_mask"].shape[1] # int = (n_cmps + sos + seq_len) - seq_len
        attn = F.pad(
            data["output_attn_mask"], 
            (npad, 0), 
            value=1,
        ) # shape (batch_size, n_cmps + sos + seq_len)
        preds = self.model(inputs_embeds=out_embs, attention_mask=attn).logits # shape [batch_size, n_cmps + sos + seq_len, vocab_size]
        preds = preds[:,cmpr.shape[1]:-1] # shape [batch_size, seq_len, vocab_size]

        return preds



    def infer(self):
        pass

    def causal_lm(self):
        pass
