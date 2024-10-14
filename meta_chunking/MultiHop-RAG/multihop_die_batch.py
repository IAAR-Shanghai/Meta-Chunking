from transformers import AutoModelForCausalLM, AutoTokenizer
import time  
import json
model_name_or_path='Qwen2-1.5B-Instruct'  
device_map = "auto"
small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)  
small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map)  
small_model.eval()

from chunk_rag import extract_by_html2text_db_chongdie
filename = 'chunk_others/corpus_ppl_qwen15B_die_0.json'  
save_list=[]
start_time = time.time() 
with open('data/corpus/corpus.txt', 'r', encoding='utf-8') as file:  
 
    content = file.read()  
sentences_lists=content.split('\n')
# print(len(sentences_lists))
# text=sentences_lists[0]
for text in sentences_lists:
    final_chunks=extract_by_html2text_db_chongdie(text,small_model,small_tokenizer,0,'en')
    save_list=save_list+final_chunks

with open(filename, 'w') as file:
    json.dump(save_list, file)
end_time = time.time()  

execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")


# CUDA_VISIBLE_DEVICES=1 nohup python multihop_die_batch.py >> chunk_others/corpus_ppl_qwen15B_die_0.log 2>&1 &

