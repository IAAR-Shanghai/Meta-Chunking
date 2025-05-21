import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import json
import time  
from tqdm import tqdm
                    # 
model_name_or_path= 'model/Qwen2-7B-Instruct'      
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map) #.to(small_device)  
model.eval()
print(model_name_or_path)

import torch.nn.functional as F
def get_prob_subtract(model,tokenizer,content,chunk):
    query='''根据提供的原始文本和从中分割出来的一个文本块，请判断该文本块是否存在缺失必要信息的情况，导致信息表述不太准确。该任务是利用原始文本内容中被明确阐述的内容，判断给定文本块是否需要补充必要信息。对于文本块未涉及到的内容不需要补充。
直接回复是或否，不要包含其他任何内容，也不要用引号、反引号或其他分隔符括住你的回复。

原始文本内容：{}

文本块：{}'''.format(content,chunk)
    prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    input_ids=prompt_ids
    output_ids = tokenizer.encode(['是','否'], return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        token_probs = F.softmax(next_token_logits, dim=-1)
    next_token_id_0 = output_ids[:, 0].unsqueeze(0)
    next_token_prob_0 = token_probs[:, next_token_id_0].item()      
    next_token_id_1 = output_ids[:, 1].unsqueeze(0)
    next_token_prob_1 = token_probs[:, next_token_id_1].item()  
    prob_subtract=next_token_prob_0-next_token_prob_1

    return prob_subtract


start_time = time.time() 
save_list=[]
filename='add_shiyan/rewrite_crud/db_qa_baichuan_rewrite_yiyan.json'
print(filename)
with open(filename, 'r', encoding='utf-8') as cfile: 
    qa_data = json.load(cfile)
threshold=0
threshold_list=[]
save_list=[]
for line in tqdm(qa_data):
    if_rewrite=[]
    full_segments = line["ppl_chunks"]
    raw_corpus = line["raw_corpus"]
    for k,sentence in enumerate(full_segments):
        if k==0:
            if_rewrite.append(0)
        else:
            prob_subtract=get_prob_subtract(model,tokenizer,raw_corpus,sentence)
            threshold_list.append(prob_subtract)
            # print('222',prob_subtract)
            if prob_subtract>threshold:
                if_rewrite.append(1)
            else:
                if_rewrite.append(0)
        if len(threshold_list)>=5:
            last_ten = threshold_list 
            avg = sum(last_ten) / len(last_ten)
            threshold=avg
            print(threshold,prob_subtract,flush=True)
    save={
        "raw_corpus": line["raw_corpus"],
        "ppl_chunks": line["ppl_chunks"],
        "rewrite": line["rewrite"],
        "if_rewrite": if_rewrite
    }
    save_list.append(save)
    with open(filename.replace('.json','_judge.json'), 'w', encoding='utf-8') as file:
        json.dump(save_list, file,ensure_ascii=False, indent=4)
    
    
end_time = time.time()  
# 计算并打印执行时间  
execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")

# CUDA_VISIBLE_DEVICES=1 nohup python judge_rewrite.py >> add_shiyan/rewrite_crud/db_qa_baichuan_rewrite_yiyan.log 2>&1 &