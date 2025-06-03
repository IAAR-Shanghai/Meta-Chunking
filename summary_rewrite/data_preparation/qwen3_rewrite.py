from openai import OpenAI
import time
import json
from tqdm import tqdm
import os

model_type='Qwen-vllm'  

if model_type=='Qwen-vllm':
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name_or_path="/mnt/afs/models/hf_models/Qwen3-32B" 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=32768) 
    model = LLM(model=model_name_or_path,tensor_parallel_size=2, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 

def prompt_llm(model_type, user_prompt):
    if model_type=='Qwen-vllm':
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )
        outputs = model.generate([text], sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
        response = generated_text
        return response


def rewrite(content,onechunk):
    prompt='''根据提供的原始文本和从中分割出来的一个文本块，利用原始文本的全局信息，请为该文本块识别其可能缺失的全局信息，主要包括模糊的指代关系，专业术语或缩写缺乏解释，被切割的重要背景信息等。
该任务是利用原始文本内容中被明确阐述的内容，为文本块补充必要信息。对于文本块未涉及到的内容不需要补充。
直接回复你的识别结果，不要包含其他任何内容，也不要用引号、反引号或其他分隔符括住你的回复。

原始文本：{}

文本块：{}'''.format(content,onechunk)
    try:
        str_result=prompt_llm(model_type, prompt)
        return str_result
    except Exception as e:
        print('111',flush=True)
        print(f"An error occurred: {e}.")
        return "LLM thinks prompt is unsafe"

def extract_information(raw_corpus, chunk,rewrite_qwq):
    prompt='''根据提供的原始文本和从中分割出来的一个文本块，认真分析“文本块缺失信息识别结果”是否适合为文本块提供可能缺失的全局信息。我们希望其提供的信息是在原始文本内容中被明确阐述的，且对于文本块未涉及到的内容不需要补充。
按照上述要求，从“文本块缺失信息识别结果”中筛选出符合要求的信息，并按照如下格式输出：
1. 缺失项：......，补充项：......
2. 缺失项：......，补充项：......
......

直接回复你的识别结果，不要包含其他任何内容，也不要用引号、反引号或其他分隔符括住你的回复。

原始文本内容：{}

文本块：{}

文本块缺失信息识别结果：{}'''.format(raw_corpus, chunk,rewrite_qwq)
    try:
        str_result=prompt_llm(model_type, prompt)
        return str_result
    except Exception as e:
        print('222',flush=True)
        print(f"An error occurred: {e}.")
        return "LLM thinks prompt is unsafe"

def rewrite2(chunk,extract_info):
    prompt='''根据提供的文本块缺失信息识别结果和对应的文本块，对文本块进行重写优化，确保补充的信息自然融入文本。你必须遵守以下4个条件：
1. 在适当的位置引入缺失信息。
2. 确保补充的信息与原文风格一致，过渡自然，不影响原有语句的表达效果。
3. 输出格式应包含完整的、经过优化后的文本块。
4. 直接回复所需的内容，不要包含任何其他内容，也不要用引号、反引号或其他分隔符括住你的回复。

文本块缺失信息识别结果：
{}

文本块：{}'''.format(extract_info,chunk)
    try:
        str_result=prompt_llm(model_type, prompt)
        return str_result
    except Exception as e:
        print('333',flush=True)
        print(f"An error occurred: {e}.")
        return "LLM thinks prompt is unsafe"


with open('rewrite/rewrite_raw_corpus.json', 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)


save_list=[]
for qa in tqdm(qa_data):
    reflection=rewrite(qa["raw_corpus"],qa["gpt_output"][0])
    if reflection == "LLM thinks prompt is unsafe":
        json_str = json.dumps(qa)
        with open('rewrite/nochunk.jsonl', 'a',encoding='utf-8') as file:
            file.write(json_str + '\n')
    else:
        reflection_tmp=reflection.split('</think>')[-1].strip()
        refinement=extract_information(qa["raw_corpus"],qa["gpt_output"][0],reflection_tmp)
        if refinement == "LLM thinks prompt is unsafe":
            json_str = json.dumps(qa)
            with open('rewrite/nochunk.jsonl', 'a',encoding='utf-8') as file:
                file.write(json_str + '\n')
        else:
            refinement_tmp=refinement.split('</think>')[-1].strip()
            completion=rewrite2(qa["gpt_output"][0],refinement_tmp)
            if completion == "LLM thinks prompt is unsafe":
                json_str = json.dumps(qa)
                with open('rewrite/nochunk.jsonl', 'a',encoding='utf-8') as file:
                    file.write(json_str + '\n')     
            else:
                save = {}
                save['raw_corpus'] = qa["raw_corpus"]
                save['gpt_output'] = qa["gpt_output"][0]
                save['reflection'] = reflection
                save['reflection_tmp'] = reflection_tmp
                save['refinement'] = refinement
                save['refinement_tmp'] = refinement_tmp
                save['completion'] = completion
                save['completion_tmp'] = completion.split('</think>')[-1].strip()
                save_list.append(save)  
                    
                with open('rewrite/rewrite_train_corpus.json', 'w', encoding='utf-8') as sfile:
                    json.dump(save_list, sfile, ensure_ascii=False, indent=4)


# CUDA_VISIBLE_DEVICES=6,7 nohup python qwen3_rewrite.py >> rewrite/rewrite_train_corpus.log 2>&1 &


