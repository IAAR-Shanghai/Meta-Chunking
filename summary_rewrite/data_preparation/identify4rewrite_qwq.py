from openai import OpenAI
import time
import json
from tqdm import tqdm
import os

model_type='qwq_api'  # qwq_api Qwen-vllm GPT

if model_type=='Qwen-vllm':
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name_or_path="model/QwQ-32B" 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, repetition_penalty=1.05, max_tokens=8192) 
    model = LLM(model=model_name_or_path,tensor_parallel_size=2, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 

def prompt_llm(model_type, user_prompt):
    if model_type == "GPT":
        try:
            client = OpenAI(
                api_key='',
                base_url="",
            )
            completion = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=4096,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}
                ])
            return completion.choices[0].message.content
        except Exception as e:
            print('111',flush=True)
            print(f"An error occurred: {e}.")
            return "GPT thinks prompt is unsafe"
    elif model_type=='Qwen-vllm':
        messages = [
            {"role": "user", "content": user_prompt}
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
    elif model_type=='qwq_api':
        client = OpenAI(
            # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
            api_key = '',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        reasoning_content = ""  # 定义完整思考过程
        answer_content = ""     # 定义完整回复
        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model="qwq-plus-latest",  # 此处以 qwq-32b 为例，可按需更换模型名称
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6, 
            top_p=0.95,
            # QwQ 模型仅支持流式输出方式调用
            stream=True,
            # 解除以下注释会在最后一个chunk返回Token使用量
            stream_options={
                "include_usage": True
            }
        )

        for chunk in completion:
            # 如果chunk.choices为空，则打印usage
            if not chunk.choices:
                print("\nUsage:",chunk.usage,flush=True)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    # print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    # # 开始回复
                    # if delta.content != "" and is_answering is False:
                    #     print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    #     is_answering = True
                    # # 打印回复过程
                    # print(delta.content, end='', flush=True)
                    answer_content += delta.content
        return reasoning_content+'</think>'+answer_content


def rewrite(content,onechunk):
    prompt='''根据提供的原始文本和从中分割出来的一个文本块，利用原始文本的全局信息，请为该文本块识别其可能缺失的全局信息，主要包括模糊的指代关系， 专业术语或缩写缺乏解释，被切割的重要背景信息等。
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
        return "GPT thinks prompt is unsafe"



with open('yiyan/rewrite_raw_corpus.json', 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)

with open('qwq/rewrite_train_corpus.json', 'r', encoding='utf-8') as file:  
    save_list = json.load(file)
# save_list=[]
for qa in tqdm(qa_data):
    raw_gpt_output=rewrite(qa["raw_corpus"],qa["gpt_output"][0])
    if raw_gpt_output == "GPT thinks prompt is unsafe":
        json_str = json.dumps(qa)
        with open('qwq/nochunk.jsonl', 'a',encoding='utf-8') as file:
            file.write(json_str + '\n')
    else:
        save = {}
        save['raw_corpus'] = qa["raw_corpus"]
        save['gpt_output'] = qa["gpt_output"][0]
        save['rewrite_qwq'] = raw_gpt_output
        save_list.append(save)  
            
        with open('qwq/rewrite_train_corpus.json', 'w', encoding='utf-8') as sfile:
            json.dump(save_list, sfile, ensure_ascii=False, indent=4)


# CUDA_VISIBLE_DEVICES=6,7 nohup python identify4rewrite_qwq.py >> qwq/rewrite_train_corpus.log 2>&1 &


# nohup python identify4rewrite_qwq.py >> qwq/rewrite_train_corpus.log 2>&1 &


