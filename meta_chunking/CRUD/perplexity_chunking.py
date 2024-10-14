import torch
import math 
import numpy as np 
import torch.nn.functional as F
from typing import List, Optional, Tuple


class Chunking:
    def __init__(self, model, tokenizer) -> None:
        self.model=model
        self.tokenizer=tokenizer
        
    def get_ppl(self,probabilities):
        ppl = 0
        for p in probabilities:
            ppl += math.log(p)
        ppl = math.exp(-ppl / len(probabilities))
        return ppl
    
    def get_ppl_batch(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None
    ):
        past_length = 0
        if end is None:
            end = input_ids.shape[1]
        with torch.no_grad():
            response =self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values
        shift_logits = response.logits[..., :-1, :].contiguous()  
        shift_labels = input_ids[..., past_length + 1 : end].contiguous()  
        # Flatten the tokens
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)  
        res = loss
        return (res, past_key_values) if return_kv else res

    def get_ppl_for_next(self,first_sentence,next_sentence):
        tokenized_text_1 = self.tokenizer(first_sentence, return_tensors="pt", add_special_tokens=False)
        tokenized_text_2 = self.tokenizer(next_sentence, return_tensors="pt", add_special_tokens=False)
        input_ids=torch.cat([tokenized_text_1["input_ids"].to(self.model.device),tokenized_text_2["input_ids"].to(self.model.device)],dim=-1)
        attention_mask = torch.cat([tokenized_text_1["attention_mask"].to(self.model.device),tokenized_text_2["attention_mask"].to(self.model.device)],dim=-1)
        with torch.no_grad():
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = response.past_key_values
        past_length=tokenized_text_1["input_ids"].to(self.model.device).shape[1]
        shift_logits = response.logits[..., past_length-1:-1, :].contiguous()  
        shift_labels = input_ids[..., past_length : ].contiguous()  
        active = (attention_mask[:, past_length:] == 1).view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)  
        res = loss.sum().item()
        return (res, past_key_values,shift_labels.shape[1])
    