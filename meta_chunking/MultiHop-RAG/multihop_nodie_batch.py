from transformers import AutoModelForCausalLM, AutoTokenizer
import time  
import torch
model_name_or_path='internlm2-chat-1_8b'  
device_map = "auto"
small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)  
small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,torch_dtype=torch.float32,device_map=device_map) #.to(small_device)  
small_model.eval()

from chunk_rag import extract_by_html2text_db_nolist

filename = 'data/corpus/corpus_nodie_internlm2_18B_00'  
start_time = time.time() 
with open('data/corpus/corpus.txt', 'r', encoding='utf-8') as file:  

    content = file.read()  
sentences_lists=content.split('\n')
# print(len(sentences_lists))
# text=sentences_lists[0]
for text in sentences_lists:
    final_chunks=extract_by_html2text_db_nolist(text,small_model,small_tokenizer,0,'en')
    # print(final_chunks)
    with open(filename, 'a', encoding='utf-8') as file:  
        file.write('\n'.join(final_chunks)+ '\n')
    
end_time = time.time()  

execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")
# CUDA_VISIBLE_DEVICES=0 nohup python multihop_nodie_batch.py >> data/corpus/corpus_nodie_internlm2_18B_00.log 2>&1 &


