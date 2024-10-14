# import json
# from tqdm import tqdm
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

# model_name_or_path="Qwen2-72B-Instruct" 
# # Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# # Pass the default decoding hyperparameters of Qwen2-7B-Instruct
# # max_tokens is for the maximum length for generation.
# sampling_params = SamplingParams(temperature=0.1, top_p=0.5, repetition_penalty=1.05, max_tokens=8192)

# # Input the model name or path. Can be GPTQ or AWQ models.
# llm = LLM(model=model_name_or_path,tensor_parallel_size=4, pipeline_parallel_size=1) 

prompt = '''You are an expert in text segmentation, tasked with dividing given text into blocks. You must adhere to the following four conditions:
1. Aim to keep each block around 128 English words in length.
2. Segment the text based solely on its logical and semantic structures.
3. Do not alter the original vocabulary or structure of the text.
4. Do not add any new words or symbols.
By solely determining the boundaries for text segmentation, divide the original text into blocks and output them individually, separated by a clear delimiter '--- Block Separator ---'. Do not output any other explanations. If you understand, please proceed to segment the following text into blocks: '''

# save_filename='data/llm_2wikimqa.json'
# with open('data/2wikimqa.json', 'r', encoding='utf-8') as file:  
#     chunk_data = json.load(file)
# llm_response=[]
# llm_blocks=[]
# i=1
# for one_chunk in tqdm(chunk_data):
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt+one_chunk}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     # text='''<|im_start|>system
#     # You are a helpful assistant.<|im_end|>
#     # <|im_start|>user\n'''+prompt+one_chunk+'\n<|im_start|>assistant'
#     print('111111111111',len(text.split()))
#     # print(text)
#     outputs = llm.generate([text], sampling_params)

#     # Print the outputs.
#     for output in outputs:
#         # prompt = output.prompt
#         generated_text = output.outputs[0].text
#         # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#         # print('111',generated_text)
    
#     response = generated_text
#     blocks = response.split("--- Block Separator ---")  
#     llm_response.append(response)
#     llm_blocks+=blocks
#     with open('data/2wikimqa_llm_response.json', 'w') as file:
#         json.dump(llm_response, file)
#     with open('data/tmp_llm.json', 'w') as file:
#         json.dump(llm_blocks, file)
#     # if i<5:
#     #     i+=1
#     # else:
#     #     break
# with open(save_filename, 'w') as file:
#     json.dump(llm_blocks, file)
    
# # CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python second_chunk_vllm.py >> logs/llm_2wikimqa.log 2>&1 &







# Chinese chunking
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name_or_path="Qwen2-72B-Instruct" 
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Pass the default decoding hyperparameters of Qwen2-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.1, top_p=0.5, repetition_penalty=1.05, max_tokens=8192)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_name_or_path,tensor_parallel_size=4, pipeline_parallel_size=1) 

prompt = '''你是一位文本分块专家，将给定的文本进行分块处理。我需要你遵守以下4个条件：
1. 尽量使每个分块的大小保持在190个汉字左右。
2. 只能按照逻辑结构和语义结构进行文本分块。
3. 不要改变原文的词汇和结构。
4. 不要添加新的词汇或符号。
通过仅判断文本分块边界的方式，对原文进行文本分块，并逐个输出分块好的文本，分块之间用‘---分块分隔符---’清晰分隔，其他任何解释都不要输出。如果你理解了，请对以下文本进行分块：'''

save_filename='data/llm_multifieldqa_zh.json'
with open('data/multifieldqa_zh.json', 'r', encoding='utf-8') as file:  
    chunk_data = json.load(file)
llm_response=[]
llm_blocks=[]
i=1
for one_chunk in tqdm(chunk_data):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt+one_chunk}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # text='''<|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user\n'''+prompt+one_chunk+'\n<|im_start|>assistant'
    print('111111111111',len(text))
    # print(text)
    outputs = llm.generate([text], sampling_params)

    # Print the outputs.
    for output in outputs:
        # prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # print('111',generated_text)
    
    response = generated_text
    blocks = response.split("---分块分隔符---")  
    llm_response.append(response)
    llm_blocks+=blocks
    with open('data/multifieldqa_zh_llm_response.json', 'w', encoding='utf-8') as file:
        json.dump(llm_response, file,ensure_ascii=False)
    with open('data/tmp_llm.json', 'w', encoding='utf-8') as file:
        json.dump(llm_blocks, file,ensure_ascii=False)
    # if i<5:
    #     i+=1
    # else:
    #     break
with open(save_filename, 'w', encoding='utf-8') as file:
    json.dump(llm_blocks, file,ensure_ascii=False)
    
# CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python second_chunk_vllm.py >> logs/llm_multifieldqa_zh.log 2>&1 &