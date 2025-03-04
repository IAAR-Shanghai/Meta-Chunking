from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import time  
import json
import torch
from tqdm import tqdm
                   
model_name_or_path='internlm3-8b-instruct'
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map)
model.eval()

## jinaai/jina-reranker-v2-base-multilingual
# from transformers import AutoModelForSequenceClassification
# model_sim = AutoModelForSequenceClassification.from_pretrained(
#     'BAAI/bge-base-zh-v1.5',
#     torch_dtype="auto",
#     trust_remote_code=True,
# )
# model_sim.to('cuda') # or 'cpu' if no GPU is available
# model_sim.eval()
# def get_simscore_for_next(model,first_sentence,next_sentence):
#     query = next_sentence
#     documents = [first_sentence]

#     # construct sentence pairs
#     sentence_pairs = [[query, doc] for doc in documents]
#     scores = model.compute_score(sentence_pairs, max_length=1024)
#     return scores


### bge-base-zh-v1.5
model_name_or_path='BAAI/all-MiniLM-L6-v2'
from sentence_transformers import SentenceTransformer
model_sim = SentenceTransformer(model_name_or_path)
def get_simscore_for_next(model,first_sentence,next_sentence):
    queries = [next_sentence]
    passages = [first_sentence]
    # instruction = "为这个句子生成表示以用于检索相关文章："

    q_embeddings = model.encode([q for q in queries], normalize_embeddings=True)
    p_embeddings = model.encode(passages, normalize_embeddings=True)
    scores = q_embeddings @ p_embeddings.T
    return scores.item()


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
    shift_logits = response.logits[..., past_length-1:-1, :].contiguous()  #模型的输出logits（即预测的类别分数）
    shift_labels = input_ids[..., past_length : ].contiguous()  #真实的目标标签（即输入ID中的下一个词）。现实中的值
    active = (attention_mask[:, past_length:] == 1).view(-1)
    active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
    active_labels = shift_labels.view(-1)[active]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(active_logits, active_labels)  #使用交叉熵损失函数计算logits和标签之间的损失。
    res = loss.mean().item()
    return res



data_path='merge_data/db_qa_semantic_68.json'
save_file='mergechunk_eval3/db_qa_semantic_68.json'

print('--'*20,data_path)

with open(data_path, 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)
start_time = time.time() 
all_data_eval=[]
sim=[]
ppl=[]
for i in tqdm(range(len(qa_data) - 1)):
    try:
        text1 = qa_data[i]
        text2 = qa_data[i + 1]
        save={}
        gen_question_sim=get_simscore_for_next(model_sim,text1,text2)
        gen_question_ppl_1=get_ppl_for_next(model,tokenizer,' ',text2)
        gen_question_ppl_2=get_ppl_for_next(model,tokenizer,text1,text2)
        
        save['no_semantic_similarity'] = 1-gen_question_sim
        sim.append(1-gen_question_sim)
        save['no_relative_perplexity'] = gen_question_ppl_2/gen_question_ppl_1
        ppl.append(gen_question_ppl_2/gen_question_ppl_1)
        all_data_eval.append(save)
        print('no_semantic_similarity_avg: ',sum(sim)/len(sim),'\n','no_relative_perplexity_avg: ',sum(ppl)/len(ppl) ,flush=True)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

with open(save_file, 'w', encoding='utf-8') as sfile:
    json.dump(all_data_eval, sfile, ensure_ascii=False, indent=4)

print('--'*20,data_path,"Final results are: ")
print('no_semantic_similarity_avg: ',sum(sim)/len(sim),'\n','no_relative_perplexity_avg: ',sum(ppl)/len(ppl) ,flush=True)
    
end_time = time.time()  
# 计算并打印执行时间  
execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")

# CUDA_VISIBLE_DEVICES=2 nohup python chunk_eval.py >> mergechunk_eval3/db_qa_semantic_68.log 2>&1 &