from openai import OpenAI
import time
import json
from tqdm import tqdm
import jieba
model_type='Qwen-vllm'  # GPT  GLM  Qwen-vllm

if model_type=='Qwen-vllm':
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name_or_path="model/Qwen2.5-72B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    sampling_params = SamplingParams(temperature=0.1, top_p=0.1, repetition_penalty=1.05, max_tokens=4096) #1280
    model = LLM(model=model_name_or_path,tensor_parallel_size=4, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.8) 



def prompt_llm(model_type, user_prompt):
    if model_type == "GPT":
        try:
            client = OpenAI(
                api_key='',
                base_url="",
            )
            completion = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=8192,
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
    elif model_type == "GLM":
        from zhipuai import ZhipuAI
        client = ZhipuAI()
        response_glm = client.chat.completions.create(
                model='glm-4-plus',  
                messages= [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt},
                    ],
            )
        ans_glm=response_glm.choices[0].message.content
        return ans_glm
    elif model_type=='Qwen-vllm':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
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

def chunking_by_llm(model_type, content):
    system_prompt='''这是一个文本分块任务，你是一名文本分割方面的专家，负责将给定的文本分割成文本块。你必须遵守以下四个条件：
1. 根据文本的逻辑和语义结构对文本进行分割，使得每个文本块都具有完整的逻辑表达。
2. 避免文本块过短，在识别内容转换与分块长度之间取得良好平衡。
3. 不要改变文本的原始词汇或内容。
4. 不要添加任何新词或符号。
如果你理解，请将以下文本分割成文本块，他们之间通过“\n---\n”来分隔。输出完整的分块结果，不能够省略。

文档内容：{}

分割好的文本块为：'''.format(content)
    # print(system_prompt)
    gpt_output=prompt_llm(model_type,system_prompt)
    raw_gpt_output=gpt_output
    
    save = {}
    if gpt_output == "GPT thinks prompt is unsafe":
        with open('train_GPT/nochunk.txt', 'a', encoding='utf-8') as file:
            file.write(content+"\n\n\n")
    elif not isinstance(gpt_output ,str):
        with open('train_GPT/nochunk.txt', 'a', encoding='utf-8') as file:
            file.write(content+"\n\n\n")
    else:
        gpt_output=gpt_output.split('\n---\n')
        gpt_output=[i.strip() for i in gpt_output if i.strip() != '']
        
        save['raw_corpus'] = content
        save['gpt_output'] = gpt_output
        save['raw_gpt_output'] = raw_gpt_output
    
    return save

def split_text_by_punctuation(text,language,target_size): 
    if language=='zh':
        sentences = jieba.cut(text, cut_all=False)  
        sentences_list = list(sentences)  
        sentences = []  
        temp_sentence = ""  
        for word in sentences_list:  
            if word in ["。", "！", "？","；"]:  
                sentences.append(temp_sentence.strip()+word)  
                temp_sentence = ""  
            else:  
                temp_sentence += word  
        if temp_sentence:   
            sentences.append(temp_sentence.strip())  
        
        merged_paragraphs = []  
        current_paragraph = "" 
        for paragraph in sentences:  
            # 检查如果当前段落加上新段落是否超过目标大小  
            if len(current_paragraph) + len(paragraph) <= target_size:  
                current_paragraph +=paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)  # 添加当前合并的段落到结果列表  
                current_paragraph = paragraph  # 重置当前段落为新段落  
        if current_paragraph:  
            merged_paragraphs.append(current_paragraph)  
        return merged_paragraphs

with open('chunking/data/webcpm_oe.txt', 'r', encoding='utf-8') as file:
    webcpm_content = file.read()  # 读取整个文件内容为一个字符串

qa_data=split_text_by_punctuation(webcpm_content,'zh',512)

save_list=[]  
start_time = time.time()           
for text in tqdm(qa_data):
    save=chunking_by_llm(model_type,text)   
    save_list.append(save)
    with open('chunking/webcpm_oe/webcpm_qwen25_72B.json', 'w', encoding='utf-8') as sfile:
        json.dump(save_list, sfile, ensure_ascii=False, indent=4)

end_time = time.time() 
execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")


# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python chunk_gpt.py >> chunking/webcpm_oe/webcpm_qwen25_72B.log 2>&1 &
