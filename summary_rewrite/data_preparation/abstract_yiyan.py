
import time
import json
from tqdm import tqdm
import requests


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

def abstract_by_llm( raw_corpus,chunk):
    system_prompt='''根据提供的原始文本和从中分割出来的一个文本块，利用原始文本的全局信息，请为该文本块生成一个补充全局信息的附加摘要。你必须遵守以下4个条件：
1. 附加摘要的语言应简洁，1-2句话能够概括。
2. 第一句话能够准确补充文本块缺失的全局内容，以确保文本块内容的完整性和清晰性，主要包括为模糊的指代关系进行补充， 解释专业术语或缩写，补充被切割的重要背景信息等。
3. 第二句话能够描述文本块的主要内容。
4. 如果文本块不需要补充全局信息，则只生成第二句话的内容，即只生成文本块的摘要。

直接回复所需的附加摘要，不要包含任何其他细节，也不要用引号、反引号或其他分隔符括住你的回复。

原始文本内容：{}

文本块：{}'''.format(raw_corpus,chunk)
    # print(system_prompt)
    try:
        str_result=get_yiyan(system_prompt)
        str_chunk_result = json.loads(str_result)["result"]
        return str_chunk_result
    except Exception as e:
        print('111',flush=True)
        print(f"An error occurred: {e}.")
        return "GPT thinks prompt is unsafe"

def condense_by_llm(abstract):
    system_prompt='''文本内容：{}

请将上述内容概括为一句话，确保这句话能够准确、全面地反映原文的主要信息。'''.format(abstract)
    # print(system_prompt)
    try:
        str_result=get_yiyan(system_prompt)
        str_chunk_result = json.loads(str_result)["result"]
        return str_chunk_result
    except Exception as e:
        print('222',flush=True)
        print(f"An error occurred: {e}.")
        return "GPT thinks prompt is unsafe"


with open('abstract/rewrite_raw_corpus.json', 'r', encoding='utf-8') as cfile: 
    qa_data = json.load(cfile)

start_time = time.time()   
save_list=[]       
for item in tqdm(qa_data):
    raw_corpus=item["raw_corpus"]
    chunk=item["gpt_output"][0]

    abstract1=abstract_by_llm(raw_corpus, chunk)
    if abstract1 == "GPT thinks prompt is unsafe":
        json_str = json.dumps(item)
        with open('abstract2/nochunk.jsonl', 'a',encoding='utf-8') as file:
            file.write(json_str + '\n')
    else:
        condense2=condense_by_llm(abstract1)
        if condense2 == "GPT thinks prompt is unsafe":
            json_str = json.dumps(item)
            with open('abstract2/nochunk.jsonl', 'a',encoding='utf-8') as file:
                file.write(json_str + '\n')
        else:
            save = {}
            save['raw_corpus'] = raw_corpus
            save['gpt_output'] = chunk
            save['abstract'] = abstract1
            save['condense'] = condense2
            save_list.append(save)  
            
            with open('abstract2/abstract_train_corpus.json', 'w', encoding='utf-8') as sfile:
                json.dump(save_list, sfile, ensure_ascii=False, indent=4)

end_time = time.time() 
execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")


# nohup python abstract_yiyan2.py >> abstract2/abstract_train_corpus.log 2>&1 &

