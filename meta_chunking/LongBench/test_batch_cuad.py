from transformers import AutoModelForCausalLM, AutoTokenizer
import time  
import json
import torch      
model_name_or_path='internlm2-chat-1_8b'   
device_map = "auto"
small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,torch_dtype=torch.float32,device_map=device_map) #.to(small_device)  
small_model.eval()

from chunk_rag import extract_by_html2text_db_bench
filename ="chunk_otherllm_00/qasper_internlm2_18b_batch.json"  

import json  
start_time = time.time() 
save_list=[]

i=0
with open('data/qasper.jsonl', 'r', encoding='utf-8') as file:  

    for line in file:  

        data = json.loads(line)  
        # print(type(data['context']),data['input'])  
        final_chunks=extract_by_html2text_db_bench(data['context'],small_model,small_tokenizer,0,language='en')
        save_list=save_list+final_chunks
        # if i>10:
        #     break
        # else:
        #     i+=1
        # with open('tmp_1.json', 'w') as file:
        #     json.dump(save_list, file)
with open(filename, 'w') as file:
    json.dump(save_list, file)
    
end_time = time.time()  

execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")
# CUDA_VISIBLE_DEVICES=0 nohup python test_batch_cuad.py >> chunk_otherllm_00/qasper_internlm2_18b_batch.log 2>&1 &