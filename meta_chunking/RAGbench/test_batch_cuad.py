from transformers import AutoModelForCausalLM, AutoTokenizer
import time  
import json
model_name_or_path='Qwen2-7B-Instruct'  
device_map = "auto"
small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map) 
small_model.eval()

from chunk_rag import extract_by_html2text_db_bench
filename ="chunking/CUAD_qwen7B_nodie_dynamic_00.json"  

import pandas as pd  
sentences_lists=[]
df = pd.read_parquet('data/CUAD/test-00000-of-00001.parquet')  
for index, row in df.iterrows(): 
    doc_lists=row['documents']
    sentences_lists.append('\n\n'.join(doc_lists))
    
    
start_time = time.time() 
save_list=[]
i=1
for text in sentences_lists:
    final_chunks=extract_by_html2text_db_bench(text,small_model,small_tokenizer,0,language='en',past_key_values=None)
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
# CUDA_VISIBLE_DEVICES=0 nohup python test_batch_cuad.py >> chunking/CUAD_qwen7B_nodie_dynamic_00.log 2>&1 &