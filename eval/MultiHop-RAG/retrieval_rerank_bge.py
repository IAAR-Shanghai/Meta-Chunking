from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time  
import json
import torch
import os
from tqdm import tqdm

model_name_or_path='BAAI/bge-reranker-large'
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map)
model.eval()

data_path='toy_data/docs_ppl_qwen15b_corpus_nodie_top10.json'
print('--'*20)
print(data_path)

with open(data_path, 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)

def get_score_for_next(model,tokenizer,first_sentence,next_sentence):
    pairs = [[first_sentence,next_sentence]]
    # print(pairs)
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # print(scores)
    return scores.item()

start_time = time.time() 
for data in tqdm(qa_data):
    raw_paragraphs=[i["text"] for i in data["retrieval_list"]]
    gen_question=data['query']
    rrf_rank_ppl=[]
    rrf_rank_final={}
    for index, paragraph in enumerate(raw_paragraphs):
        gen_question_ppl=get_score_for_next(model,tokenizer,paragraph,gen_question)
        rrf_rank_ppl.append(gen_question_ppl)

    # Create a sorted list containing the original index and values 
    sorted_lst = sorted(enumerate(rrf_rank_ppl), key=lambda x: x[1],reverse=True)  
    # Create a dictionary to store the sorting position (ranking) of each element
    rank_dict = {value: idx + 1 for idx, (original_index, value) in enumerate(sorted_lst)}  
    for i in range(len(raw_paragraphs)):
        rrf_rank_final[i]={'sentence':raw_paragraphs[i],'similarity_ppl':[i+1,rank_dict[rrf_rank_ppl[i]]]}
    rrf_k=60
    for i,j in rrf_rank_final.items():
        j['rrf']=1.0 / (rrf_k+j['similarity_ppl'][0])+1.0 / (rrf_k+j['similarity_ppl'][1])
    rrf_rank_final = sorted(rrf_rank_final.items(), key=lambda x: x[1]['rrf'], reverse=True) 
    
    rerank_result=[]
    for i in range(len(data["retrieval_list"])):
        tmp={"text":rrf_rank_final[i][1]['sentence']}
        rerank_result.append(tmp)
    
    data["retrieval_list"]=rerank_result

first, second = os.path.splitext(data_path) 
save_file = first+'_rerank_bge'+second
with open(save_file, 'w') as json_file:
    json.dump(qa_data, json_file,indent=4)
    
end_time = time.time()  
execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")

# CUDA_VISIBLE_DEVICES=5 nohup python retrieval_rerank_bge.py >> ppl_rerank_bge.log 2>&1 &