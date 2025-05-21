import requests
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name_or_path='abstract_model/3_1'
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)          
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,torch_dtype=torch.float32,device_map=device_map) 
model.eval()

def get_trainqwen(prompt):
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

def abstract(content,onechunk):
    prompt='''根据提供的原始文本和从中分割出来的一个文本块，利用原始文本的全局信息，为该文本块生成一个补充全局信息的附加摘要。你必须遵守以下3个条件：
1. 附加摘要的语言应简洁，1句话能够概括。
2. 附加摘要能够描述文本块的主要内容。
3. 确保附加摘要包含文本块缺失的全局内容，以保证生成摘要的完整性和清晰性。主要包括为模糊的指代关系进行补充， 解释专业术语或缩写，补充被切割的重要背景信息等。

直接回复所需的附加摘要，不要包含任何其他细节，也不要用引号、反引号或其他分隔符括住你的回复。


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
        zabstract=[]
        for item in qa["ppl_chunks"]:
            if len(item)<20:
                pass
            else: 
                raw_gpt_output=abstract(qa["raw_corpus"],item)
                if raw_gpt_output == "GPT thinks prompt is unsafe":
                    zabstract.append(item)
                else:
                    zabstract.append(raw_gpt_output+'\n'+item)
        save['raw_corpus'] = qa["raw_corpus"]
        save['ppl_chunks'] = qa["ppl_chunks"]
        save['rewrite'] = zabstract
        save_list.append(save)  
        
        with open('addition_shiyan/abstract/db_qa_baichuan_abstract_yiyan.json', 'w', encoding='utf-8') as sfile:
            json.dump(save_list, sfile, ensure_ascii=False, indent=4)

        


if __name__ == '__main__':
    main()
    
# CUDA_VISIBLE_DEVICES=2 nohup python test_abstract.py >> addition_shiyan/abstract/db_qa_baichuan_abstract_yiyan.log 2>&1 &


# 