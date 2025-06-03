import time
import json
from tqdm import tqdm



model_type='Qwen-vllm'  # Qwen-vllm

if model_type=='Qwen-vllm':
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name_or_path="/mnt/afs/models/hf_models/Qwen3-32B" 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=32768) 
    model = LLM(model=model_name_or_path,tensor_parallel_size=2, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 

def prompt_llm(model_type, user_prompt,thinking=True):
    if model_type=='Qwen-vllm':
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,  # Set to False to strictly disable thinking
        )
        outputs = model.generate([text], sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
        response = generated_text
        return response

def abstract_by_llm(raw_corpus,chunk):
    user_prompt='''根据提供的原始文本和从中分割出来的一个文本块，利用原始文本的全局信息，请为该文本块生成一个补充全局信息的附加摘要。你必须遵守以下4个条件：
1. 附加摘要的语言应简洁，1-2句话能够概括。
2. 第一句话能够准确补充文本块缺失的全局内容，以确保文本块内容的完整性和清晰性，主要包括为模糊的指代关系进行补充， 解释专业术语或缩写，补充被切割的重要背景信息等。
3. 第二句话能够描述文本块的主要内容。
4. 如果文本块不需要补充全局信息，则只生成第二句话的内容，即只生成文本块的摘要。

直接回复所需的附加摘要，不要包含任何其他细节，也不要用引号、反引号或其他分隔符括住你的回复。

原始文本内容：{}

文本块：{}'''.format(raw_corpus,chunk)
    # print(system_prompt)
    try:
        str_result=prompt_llm(model_type, user_prompt)
        return str_result
    except Exception as e:
        print('111',flush=True)
        print(f"An error occurred: {e}.")
        return "LLM thinks prompt is unsafe"

def condense_by_llm(abstract):
    user_prompt='''文本内容：{}

请将上述内容概括为一句话，确保这句话能够准确、全面地反映原文的主要信息。直接回复所需的内容，不要包含其他任何内容，也不要用引号、反引号或其他分隔符括住你的回复。'''.format(abstract)
    # print(system_prompt)
    try:
        str_result=prompt_llm(model_type, user_prompt,thinking=False)
        return str_result
    except Exception as e:
        print('222',flush=True)
        print(f"An error occurred: {e}.")
        return "LLM thinks prompt is unsafe"


with open('abstract_crud/raw_corpus.json', 'r', encoding='utf-8') as cfile: 
    qa_data = json.load(cfile)



start_time = time.time()   
save_list=[]       
for item in tqdm(qa_data):
    raw_corpus=item["raw_corpus"]
    chunk=item["gpt_output"]

    abstract1=abstract_by_llm(raw_corpus, chunk)
    if abstract1 == "LLM thinks prompt is unsafe":
        json_str = json.dumps(item)
        with open('abstract_crud/nochunk.jsonl', 'a',encoding='utf-8') as file:
            file.write(json_str + '\n')
    else:
        abstract1_tmp=abstract1.split('</think>')[-1].strip()
        condense2=condense_by_llm(abstract1_tmp)
        if condense2 == "LLM thinks prompt is unsafe":
            json_str = json.dumps(item)
            with open('abstract_crud/nochunk.jsonl', 'a',encoding='utf-8') as file:
                file.write(json_str + '\n')
        else:
            save = {}
            save['raw_corpus'] = raw_corpus
            save['gpt_output'] = chunk
            save['abstract'] = abstract1
            save['abstract_tmp'] = abstract1_tmp
            save['condense'] = condense2
            save_list.append(save)  
            
            with open('abstract_crud/abstract_train_corpus.json', 'w', encoding='utf-8') as sfile:
                json.dump(save_list, sfile, ensure_ascii=False, indent=4)

end_time = time.time() 
execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")


# CUDA_VISIBLE_DEVICES=6,7 nohup python qwen3_abstract.py >> abstract_crud/abstract_train_corpus.log 2>&1 &