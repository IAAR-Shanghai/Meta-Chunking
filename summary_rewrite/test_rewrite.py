import requests
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_type='Qwen-vllm'  # qwen Qwen-vllm 

if model_type=='Qwen-vllm':
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name_or_path="rewrite_model/3_1" 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    sampling_params = SamplingParams(temperature=0.1, top_p=0.1, repetition_penalty=1.05, max_tokens=1280) 
    model = LLM(model=model_name_or_path,tensor_parallel_size=1, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 
elif model_type=='qwen':                                                                         # float32  float16
    model_name_or_path='rewrite_model/3_1'
    device_map = "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)        
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,torch_dtype=torch.float32,device_map=device_map) 
    model.eval()

def get_trainqwen(prompt):
    if model_type=='qwen':
        messages = [
            {"role": "system", "content": " You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1280,
            temperature=0.1,
            top_p=0.1
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    elif model_type=='Qwen-vllm':
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = model.generate([text], sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
        response = generated_text
        return response

def rewrite(content,onechunk):
    prompt='''根据提供的原始文本和从中分割出来的一个文本块，利用原始文本的全局信息，对文本块中因信息缺失而模糊的地方进行补充。你必须遵守以下六个条件：
1. 为模糊的指代关系进行补充。
2. 补充被切割的重要背景信息。
3. 解释专业术语或缩写。
4. 保持原有写作风格不变。
5. 尽可能保持文本块的其他内容和表述不变。
6. 不要为文本块省略信息或添加未在原始文本中提及的内容。

如果你理解，直接回复完善好的文本块，不要输出任何其他信息，也不要用引号或其他分隔符括住你的回复。


原始文本：{}

文本块：{}'''.format(content,onechunk)
    try:
        str_result=get_trainqwen(prompt)
        return str_result
    except Exception as e:
        print('111',flush=True)
        print(f"An error occurred: {e}.")
        return "GPT thinks prompt is unsafe"


def main():
    with open('addition_shiyan/chunking/db_qa_baichuan_nodie_0_all.json', 'r', encoding='utf-8') as file:  
        qa_data = json.load(file)
    save_list=[]
    for qa in tqdm(qa_data):
        save = {}
        zrewrite=[]
        for item in qa["ppl_chunks"]:
            if len(item)<20:
                pass
            else: 
                raw_gpt_output=rewrite(qa["raw_corpus"],item)
                if raw_gpt_output == "GPT thinks prompt is unsafe":
                    zrewrite.append(item)
                else:
                    zrewrite.append(raw_gpt_output)
        save['raw_corpus'] = qa["raw_corpus"]
        save['ppl_chunks'] = qa["ppl_chunks"]
        save['rewrite'] = zrewrite
        save_list.append(save)  
        
        with open('addition_shiyan/rewrite/db_qa_baichuan_rewrite_qwqvllm16.json', 'w', encoding='utf-8') as sfile:
            json.dump(save_list, sfile, ensure_ascii=False, indent=4)

        


if __name__ == '__main__':
    main()
    
# CUDA_VISIBLE_DEVICES=0 nohup python test_rewrite.py >> addition_shiyan/rewrite/db_qa_baichuan_rewrite_qwqvllm16.log 2>&1 &


