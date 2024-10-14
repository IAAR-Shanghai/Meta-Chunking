from transformers import AutoModelForCausalLM, AutoTokenizer
import time  
import json
model_name_or_path= 'Qwen2-1.5B-Instruct'   
device_map = "auto"
small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map) #.to(small_device)  
small_model.eval()

from chunk_rag import extract_by_html2text_db_nolist
filename ="chunk_qwen72B_other_thrd/qasper_dynamic_qwen15B_02.json"  

import json  
start_time = time.time() 
save_list=[]
 
i=0
with open('data/qasper.jsonl', 'r', encoding='utf-8') as file:  
    for line in file:  
        data = json.loads(line)  
        # print(type(data['context']),data['input'])  
        final_chunks=extract_by_html2text_db_nolist(data['context'],small_model,small_tokenizer,0.2,language='en')  
        save_list=save_list+final_chunks
        # if i>10:
        #     break
        # else:
        #     i+=1
with open(filename, 'w') as file:
    json.dump(save_list, file)
    
end_time = time.time()  

execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")
# CUDA_VISIBLE_DEVICES=2 nohup python test_batch1_forqwen72B.py >> chunk_qwen72B_other_thrd/qasper_dynamic_qwen15B_02.log 2>&1 &