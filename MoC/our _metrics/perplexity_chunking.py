import torch


class Chunking:
    def __init__(self, model, tokenizer) -> None:
        self.model=model
        self.tokenizer=tokenizer
        
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
        shift_logits = response.logits[..., past_length-1:-1, :].contiguous()  #模型的输出logits（即预测的类别分数）
        shift_labels = input_ids[..., past_length : ].contiguous()  #真实的目标标签（即输入ID中的下一个词）。现实中的值
        active = (attention_mask[:, past_length:] == 1).view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)  #使用交叉熵损失函数计算logits和标签之间的损失。
        res = loss.sum().item()
        return (res, past_key_values,shift_labels.shape[1])

