# Used for long text, dynamic transformation threshold
from transformers import AutoModelForCausalLM, AutoTokenizer
import time  
import json
model_name_or_path='Qwen2-1.5B-Instruct' 
device_map = "auto"
small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map) 
small_model.eval()

from chunk_rag import extract_by_html2text_db_dynamic_batch
filename ="chunk_qwen72B_threshold/narrativeqa_dynamic_qwen15B_thrd_min_minall_batch.json"  

import json  
start_time = time.time() 
save_list=[]
 
i=0
threshold_zlist=[]
threshold=0
with open('data/narrativeqa.jsonl', 'r', encoding='utf-8') as file:  

    for line in file:  
        data = json.loads(line)  
        # print(type(data['context']),data['input'])        # Pay attention to chunk language
        final_chunks,threshold,threshold_zlist=extract_by_html2text_db_dynamic_batch(data['context'],small_model,small_tokenizer,threshold,threshold_zlist,language='en')   #修改
        save_list=save_list+final_chunks
        print('threshold',threshold,threshold_zlist[-10:])
        # if i>10:
        #     break
        # else:
        #     i+=1
with open(filename, 'w') as file:
    json.dump(save_list, file)
    
end_time = time.time()  

execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")
# CUDA_VISIBLE_DEVICES=6 nohup python test_batch1_threshold.py >> chunk_qwen72B_threshold/narrativeqa_dynamic_qwen15B_thrd_min_minall_batch.log 2>&1 &