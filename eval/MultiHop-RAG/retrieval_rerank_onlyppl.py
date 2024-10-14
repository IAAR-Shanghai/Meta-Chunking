from transformers import AutoModelForCausalLM, AutoTokenizer
import time  
import json
import torch
import os
from tqdm import tqdm

model_name_or_path='Qwen2-1.5B-Instruct' 
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map)
model.eval()

data_path='toy_data/docs_ppl_qwen15b_corpus_nodie_top10.json'
print('--'*20)
print(data_path)

with open(data_path, 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)

def get_ppl_for_next(model,tokenizer,first_sentence,next_sentence):
    tokenized_text_1 = tokenizer(first_sentence, return_tensors="pt", add_special_tokens=False)
    tokenized_text_2 = tokenizer(next_sentence, return_tensors="pt", add_special_tokens=False)
    input_ids=torch.cat([tokenized_text_1["input_ids"].to(model.device),tokenized_text_2["input_ids"].to(model.device)],dim=-1)
    attention_mask = torch.cat([tokenized_text_1["attention_mask"].to(model.device),tokenized_text_2["attention_mask"].to(model.device)],dim=-1)
    with torch.no_grad():
        response = model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        # past_key_values = response.past_key_values
    past_length=tokenized_text_1["input_ids"].to(model.device).shape[1]
    shift_logits = response.logits[..., past_length-1:-1, :].contiguous()  
    shift_labels = input_ids[..., past_length : ].contiguous()  
    active = (attention_mask[:, past_length:] == 1).view(-1)
    active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
    active_labels = shift_labels.view(-1)[active]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(active_logits, active_labels)  
    res = loss.mean().item()
    return res

start_time = time.time() 
for data in tqdm(qa_data):
    raw_paragraphs=[i["text"] for i in data["retrieval_list"]]
    gen_question=data['query']
    rrf_rank_ppl=[]
    rrf_rank_final={}
    for index, paragraph in enumerate(raw_paragraphs):
        gen_question_ppl=get_ppl_for_next(model,tokenizer,paragraph+'\nBased on the above content, generate a question:',gen_question)
        rrf_rank_ppl.append(gen_question_ppl)


    sorted_lst = sorted(enumerate(rrf_rank_ppl), key=lambda x: x[1])  

    rerank_result=[]
    for i in range(len(data["retrieval_list"])):
        tmp={"text":raw_paragraphs[sorted_lst[i][0]]}
        rerank_result.append(tmp)
    
    data["retrieval_list"]=rerank_result

first, second = os.path.splitext(data_path) 
save_file = first+'_rerank_onlyqwen15B_2'+second
with open(save_file, 'w') as json_file:
    json.dump(qa_data, json_file,indent=4)
    
end_time = time.time()  

execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")

# CUDA_VISIBLE_DEVICES=3 nohup python retrieval_rerank_onlyppl.py >> ppl_rerank_onlyqwen15B_2.log 2>&1 &