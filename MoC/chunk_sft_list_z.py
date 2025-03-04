from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
import time  
import json
import re
from nltk.tokenize import sent_tokenize
import jieba

model_name_or_path= 'model/firefly-qwen25-15b-sft-full-all-M-3'   
print('-'*10,model_name_or_path)
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map) 
model.eval()

model_name_or_path_1= 'model/Qwen2.5-7B-Instruct'   
device_map = "auto"
tokenizer_extract = AutoTokenizer.from_pretrained(model_name_or_path_1,trust_remote_code=True)  
model_extract = AutoModelForCausalLM.from_pretrained(model_name_or_path_1, trust_remote_code=True,device_map=device_map) 
model_extract.eval()

def prompt_llm(model_type,user_prompt,model,tokenizer,temperature=0.1,top_p=0.1,max_new_tokens=1280):
    if model_type=='Qwen':
        messages = [
            {"role": "system", "content": " You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,  #后续可能要调大
            temperature=temperature,
            top_p=top_p
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

def chunking_by_llmsft(model_type,text,model,tokenizer,temperature=0.1,top_p=0.1):
    user_prompt='''这是一个文本分块任务，你是一名文本分割方面的专家，负责将给定的文本分割成文本块。你必须遵守以下四个条件：
1. 将内容相关的若干个连续句子组合成文本块，使得每个文本块都具有完整的逻辑表达。
2. 避免文本块过短，在识别内容转换与分块长度之间取得良好平衡。
3. 输出的分块结果是一个列表格式，其中的每个元素表示文档中的一个文本块。
4. 其中输出的每个文本块由文本块开头和结尾几个字符组成，中间内容由“[MASK]”来代替，输出格式如下：
[
    "文本块1的开头几个字符[MASK]文本块1的结尾几个字符",
    ......
]

如果你理解，请将以下文本分割成文本块，并通过上述要求输出列表格式。

文档内容：'''
    prompt=user_prompt+text+'\n\n分割好的文本块的列表格式输出：'
    response=prompt_llm(model_type,prompt,model,tokenizer,temperature,top_p)
    return response

def is_valid_json(json_str):
    """检查给定的字符串是否是有效的格式"""
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False

def split_text_by_punctuation(text,language,target_size): 
    if language=='en': 
        full_segments = sent_tokenize(text)
        merged_paragraphs = []  
        current_paragraph = "" 
        for paragraph in full_segments:  
            # 检查如果当前段落加上新段落是否超过目标大小  
            if len(current_paragraph.split()) + len(paragraph.split()) <= target_size:  
                current_paragraph +=' '+paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)  # 添加当前合并的段落到结果列表  
                current_paragraph = paragraph  # 重置当前段落为新段落  
        if current_paragraph:  
            merged_paragraphs.append(current_paragraph)  
        return merged_paragraphs
    elif language=='zh':
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


def chunking_by_llm_list(model_type, content):
    system_prompt='''这是一个文本分块任务，你是一名文本分割方面的专家，负责将给定的文本分割成文本块。你必须遵守以下四个条件：
1. 根据文本的逻辑和语义结构对文本进行分割，使得每个文本块都具有完整的逻辑表达。
2. 避免文本块过长或过短，在识别内容转换与分块长度之间取得良好平衡。
3. 不要改变文本的原始词汇或内容。
4. 不要添加任何新词或符号。
如果你理解，请将以下文本分割成文本块，他们之间严格通过“\n---\n”来分隔。

文档内容：{}

分割好的文本块为：'''.format(content)
    gpt_output=prompt_llm(model_type,system_prompt,model_extract,tokenizer_extract,max_new_tokens=2048)
    
    save = {}
    gpt_output=gpt_output.split('\n---\n')
    gpt_output=[i.strip() for i in gpt_output if i.strip() != '']
    
    save['raw_corpus'] = content
    save['gpt_output'] = ''
    save['final_chunks'] = gpt_output
    
    return save


import Levenshtein as lev  

def find_most_similar_text(reference_text, original_text):
    window_size=len(reference_text)
    # 计算给定文本与原文中每个段落之间的编辑距离
    min_distance = float('inf')
    most_similar_paragraph = None
    
    for i in range(len(original_text) - window_size + 1):
        window = original_text[i:i + window_size]
        
        # 计算与开头和结尾的编辑距离
        distance = lev.distance(window, reference_text)
        
        # 如果当前距离比最小值还小，则更新最相似的段落
        if distance < min_distance:
            min_distance = distance
            most_similar_paragraph = window

    return most_similar_paragraph, min_distance


model_type="Qwen"
filename ="chunking/Hyperpara_analysis/dureader_topk09.json"  
max_attempts = 3

root_dir = 'chunking/data/dureader'
for root, dirs, files in os.walk(root_dir):  
    for file in files:
        print('-'*10,os.path.join(root, file))
        start_time = time.time() 
        with open(os.path.join(root, file), 'r', encoding='utf-8') as zfile:
            lines = zfile.readlines()
        all_data_chunk=[]
        qa_data = [json.loads(line)["context"] for line in lines]
        save_list=[]          
        for text in tqdm(qa_data):
            merged_paragraphs=split_text_by_punctuation(text,'zh',1024)
            for zcontext in merged_paragraphs:
                chunks_json=chunking_by_llmsft(model_type,zcontext,model,tokenizer,temperature=0.1,top_p=0.9)   # 修改这里
                print('111',chunks_json,flush=True)
                
                if not is_valid_json(chunks_json):
                    for attempt in range(max_attempts):
                        chunks_json=chunking_by_llmsft(model_type,zcontext,model,tokenizer,0.5,0.5) 
                        print(f"Valid list generated on attempt {attempt + 1}",flush=True)
                        if is_valid_json(chunks_json):
                            break
                if not is_valid_json(chunks_json):
                    save=chunking_by_llm_list(model_type,zcontext)  
                else:
                    keys_list = json.loads(chunks_json)
                    new_final_chunks=[]
                    for regex_pattern in keys_list:
                        if regex_pattern.strip() !='':
            #                 regex_pattern=regex_pattern.replace('[MASK]','.*?')  #####正则匹配这修改好
            #                 try:
            #                     matches = re.findall(regex_pattern, zcontext, re.DOTALL)
            #                     new_final_chunks.append(matches[0])
            #                 except:
            #                     print('222',regex_pattern,flush=True)
            #                     extract_prompt='''正则表达式：{}
            # 您是一名文本提取专家。请根据上述正则表达式从以下文档中提取出相应信息。只输出与文档内容匹配的原文本，不输出任何解释。

            # 文档内容：{}'''.format(regex_pattern,zcontext)

            #                     match_output=prompt_llm(model_type,extract_prompt,model_extract,tokenizer_extract)
            #                     match_output.replace(regex_pattern,'')
            #                     new_final_chunks.append(match_output.strip())
                            zzz=regex_pattern
                            regex_pattern=regex_pattern.replace('[MASK]','.*?')
                            regex_pattern=regex_pattern.replace('|','\|')
                            try:
                                matches = re.findall(regex_pattern, zcontext, re.DOTALL)
                                new_final_chunks.append(matches[0])
                            except:
                                # 非模型方法实现匹配
                                try:
                                    my_regex=zzz
                                    if '[MASK]' in my_regex:
                                        # 假设我们在第一个空格处分割文本
                                        left_right = re.split(r'\[MASK\]', my_regex)
                                        tmp_left=left_right[0]
                                        tmp_right=left_right[-1]
                                        left, distance1 = find_most_similar_text(tmp_left, zcontext)
                                        right, distance2 = find_most_similar_text(tmp_right, zcontext)
                                        my_regex=left+".*?"+right
                                        my_regex=my_regex.replace('[','\[')
                                        my_regex=my_regex.replace(']','\]')
                                        my_regex=my_regex.replace('|','\|')
                                        matches = re.findall(my_regex, zcontext, re.DOTALL)
                                        # print(matches)
                                        new_final_chunks.append(matches[0])
                                    else:
                                        new_final_chunks.append(my_regex)
                                except:
                                    print([regex_pattern],flush=True)
                                    extract_prompt='''正则表达式：{}
            您是一名文本提取专家。请根据上述正则表达式从以下文档中提取出相应文本块，其中正则表达式由文本块开头和结尾几个字符和中间的省略符号组成，尽可能完整的识别并提取，保证文本块开头和结尾与正则表达式的开头和结尾匹配。只输出与文档内容匹配的原文本，不输出任何解释。

            文档内容：{}'''.format(regex_pattern,zcontext)

                                    match_output=prompt_llm(model_type,extract_prompt,model_extract,tokenizer_extract)
                                    match_output.replace(regex_pattern,'')
                                    new_final_chunks.append(match_output.strip())
                    save = {}
                    save['raw_corpus'] = zcontext
                    save['gpt_output'] = chunks_json
                    save['final_chunks'] = new_final_chunks
                
                all_data_chunk.append(save)
         
                with open(filename, 'w', encoding='utf-8') as sfile:
                    json.dump(all_data_chunk, sfile, ensure_ascii=False, indent=4)

        end_time = time.time()  
        # Calculate and print execution time
        execution_time = end_time - start_time  
        print(f"程序执行时间为: {execution_time} 秒")

# CUDA_VISIBLE_DEVICES=0 nohup python chunk_sft_list_zz.py >> chunking/Hyperpara_analysis/dureader_topk09.log 2>&1 &