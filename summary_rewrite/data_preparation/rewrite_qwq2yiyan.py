import requests
import json
import re
from tqdm import tqdm


def get_yiyan(prompt):
        
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-128k?access_token=" + ""#get_access_token()
    
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.8,
        "top_p": 0.8,
        "penalty_score": 1,
        "disable_search": True,
        "enable_citation": False,
        "response_format": "text"
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    
    return response.text


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
        str_result=get_yiyan(prompt)
        str_chunk_result = json.loads(str_result)["result"]
        return str_chunk_result
    except Exception as e:
        print('111',flush=True)
        print(f"An error occurred: {e}.")
        return "GPT thinks prompt is unsafe"


def rewrite(chunk,extract_info):
    prompt='''根据提供的文本块缺失信息识别结果和对应的文本块，对文本块进行重写优化，确保补充的信息自然融入文本。你必须遵守以下4个条件：
1. 在适当的位置引入缺失信息。
2. 确保补充的信息与原文风格一致，过渡自然，不影响原有语句的表达效果。
3. 输出格式应包含完整的、经过优化后的文本块。
4. 直接回复所需的内容，不要包含任何其他内容，也不要用引号、反引号或其他分隔符括住你的回复。

文本块缺失信息识别结果：
{}

文本块：{}'''.format(extract_info,chunk)
    try:
        str_result=get_yiyan(prompt)
        str_chunk_result = json.loads(str_result)["result"]
        return str_chunk_result
    except Exception as e:
        print('222',flush=True)
        print(f"An error occurred: {e}.")
        return "GPT thinks prompt is unsafe"



with open('qwq2yiyan/rewrite_train_corpus.json', 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)[10000:]

with open('qwq2yiyan/rewrite_train_qwq2yiyan.json', 'r', encoding='utf-8') as file:  
    save_list = json.load(file)
# save_list=[]
for item in tqdm(qa_data):
    raw_corpus=item["raw_corpus"]
    chunk=item["gpt_output"]
    rewrite_qwq=item['rewrite_qwq'].split('</think>')[-1].strip()
    
    extract_info=extract_information(raw_corpus, chunk,rewrite_qwq)
    extract_info=extract_info.strip()
    if extract_info == "GPT thinks prompt is unsafe":
        json_str = json.dumps(item)
        with open('qwq2yiyan/nochunk.jsonl', 'a',encoding='utf-8') as file:
            file.write(json_str + '\n')
    else:
        rewrite_info=rewrite(chunk,extract_info)
        if rewrite_info == "GPT thinks prompt is unsafe":
            json_str = json.dumps(item)
            with open('qwq2yiyan/nochunk.jsonl', 'a',encoding='utf-8') as file:
                file.write(json_str + '\n')
        else:
            save = {}
            save['raw_corpus'] = raw_corpus
            save['gpt_output'] = chunk
            save['rewrite_qwq'] = rewrite_qwq
            save['extract_info'] = extract_info
            save['rewrite_info'] = rewrite_info
            save_list.append(save)  
            
            with open('qwq2yiyan/rewrite_train_qwq2yiyan.json', 'w', encoding='utf-8') as sfile:
                json.dump(save_list, sfile, ensure_ascii=False, indent=4)
        

# nohup python rewrite_qwq2yiyan.py >> qwq2yiyan/rewrite_train_qwq2yiyan.log 2>&1 &
