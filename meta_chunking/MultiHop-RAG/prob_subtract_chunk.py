####Use this code for English, not GLM, only consider adjacent two sentences
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import json
from nltk.tokenize import sent_tokenize
import time  
  
model_name_or_path= 'pythia-160m'    
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map) 
model.eval()

def get_prob_subtract(model,tokenizer,sentence1,sentence2,language):
    if language=='zh':
        query='''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
        1. 将“{}”分割成“{}”与“{}”两部分；
        2. 将“{}”不进行分割，保持原形式；
        请回答1或2。'''.format(sentence1+sentence2,sentence1,sentence2,sentence1+sentence2)
        prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids=prompt_ids
        output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()      
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()  
        prob_subtract=next_token_prob_1-next_token_prob_0
    else:
        query='''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
        1. Split "{}" into "{}" and "{}" two parts;
        2. Keep "{}" unsplit in its original form;
        Please answer 1 or 2.'''.format(sentence1+' '+sentence2,sentence1,sentence2,sentence1+' '+sentence2)
        prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids=prompt_ids
        output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()      
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()  
        prob_subtract=next_token_prob_1-next_token_prob_0
    return prob_subtract


start_time = time.time() 
save_list=[]
filename='chunk_others/corpus_prob_onlytwo_pythia_160M.json'
with open('data/corpus/corpus.txt', 'r', encoding='utf-8') as file:  
 
    content = file.read()  
sentences_lists=content.split('\n')
# print(len(sentences_lists))
# text=sentences_lists[0]
for text in sentences_lists: 
    threshold=0
    threshold_list=[]

    full_segments = sent_tokenize(text)
    print('111',len(full_segments))
    tmp=''
    for i,sentence in enumerate(full_segments):
        if tmp=='':
            tmp+=sentence
        else:
            prob_subtract=get_prob_subtract(model,tokenizer,full_segments[i-1],sentence,'en')    
            threshold_list.append(prob_subtract)
            # print('222',prob_subtract)
            if prob_subtract>threshold:
                tmp+=' '+sentence
            else:
                save_list.append(tmp)
                tmp=sentence
        if len(threshold_list)>=5:
            last_ten = threshold_list[-5:]
            avg = sum(last_ten) / len(last_ten)
            threshold=avg
            print(threshold,last_ten)  
    if tmp!='':
        save_list.append(tmp)
    with open('tmp_8.json', 'w') as file:
        json.dump(save_list, file)
with open(filename, 'w') as file:
    json.dump(save_list, file)
    
end_time = time.time()  

execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")


# CUDA_VISIBLE_DEVICES=0 nohup python prob_subtract_chunk.py >> chunk_others/corpus_prob_onlytwo_pythia_160M.log 2>&1 &




# ####Use this code in English, not GLM
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch.nn.functional as F
# import json
# from nltk.tokenize import sent_tokenize
# import time  

# model_name_or_path= 'Qwen2-1.5B-Instruct'       
# device_map = "auto"
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map) #.to(small_device)  
# model.eval()

# def get_prob_subtract(model,tokenizer,sentence1,sentence2,language):
#     if language=='zh':
#         query='''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
#         1. 将“{}”分割成“{}”与“{}”两部分；
#         2. 将“{}”不进行分割，保持原形式；
#         请回答1或2。'''.format(sentence1+sentence2,sentence1,sentence2,sentence1+sentence2)
#         prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
#         prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
#         input_ids=prompt_ids
#         output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
#         with torch.no_grad():
#             outputs = model(input_ids)
#             next_token_logits = outputs.logits[:, -1, :]
#             token_probs = F.softmax(next_token_logits, dim=-1)
#         next_token_id_0 = output_ids[:, 0].unsqueeze(0)
#         next_token_prob_0 = token_probs[:, next_token_id_0].item()      
#         next_token_id_1 = output_ids[:, 1].unsqueeze(0)
#         next_token_prob_1 = token_probs[:, next_token_id_1].item()  
#         prob_subtract=next_token_prob_1-next_token_prob_0
#     else:
#         query='''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
#         1. Split "{}" into "{}" and "{}" two parts;
#         2. Keep "{}" unsplit in its original form;
#         Please answer 1 or 2.'''.format(sentence1+' '+sentence2,sentence1,sentence2,sentence1+' '+sentence2)
#         prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
#         prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
#         input_ids=prompt_ids
#         output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
#         with torch.no_grad():
#             outputs = model(input_ids)
#             next_token_logits = outputs.logits[:, -1, :]
#             token_probs = F.softmax(next_token_logits, dim=-1)
#         next_token_id_0 = output_ids[:, 0].unsqueeze(0)
#         next_token_prob_0 = token_probs[:, next_token_id_0].item()      
#         next_token_id_1 = output_ids[:, 1].unsqueeze(0)
#         next_token_prob_1 = token_probs[:, next_token_id_1].item()  
#         prob_subtract=next_token_prob_1-next_token_prob_0
#     return prob_subtract


# start_time = time.time() 
# save_list=[]
# filename='chunk_others/corpus_prob_qwen15B.json'
# with open('data/corpus/corpus.txt', 'r', encoding='utf-8') as file:  
#     content = file.read()  
# sentences_lists=content.split('\n')
# # print(len(sentences_lists))
# # text=sentences_lists[0]
# for text in sentences_lists: 
#     threshold=0
#     threshold_list=[]

#     full_segments = sent_tokenize(text)
#     print('111',len(full_segments))
#     tmp=''
#     for sentence in full_segments:
#         if tmp=='':
#             tmp+=sentence
#         else:
#             prob_subtract=get_prob_subtract(model,tokenizer,tmp,sentence,'en')    
#             threshold_list.append(prob_subtract)
#             # print('222',prob_subtract)
#             if prob_subtract>threshold:
#                 tmp+=' '+sentence
#             else:
#                 save_list.append(tmp)
#                 tmp=sentence
#         if len(threshold_list)>=5:
#             last_ten = threshold_list[-5:]  
#             avg = sum(last_ten) / len(last_ten)
#             threshold=avg
#     if tmp!='':
#         save_list.append(tmp)
#     with open('tmp_5.json', 'w') as file:
#         json.dump(save_list, file)
# with open(filename, 'w') as file:
#     json.dump(save_list, file)
    
# end_time = time.time()  

# execution_time = end_time - start_time  
# print(f"程序执行时间为: {execution_time} 秒")


# # CUDA_VISIBLE_DEVICES=4 nohup python prob_subtract_chunk.py >> chunk_others/corpus_prob_qwen15B.log 2>&1 &
