import time
import re
import pandas as pd
import time  
from nltk.tokenize import sent_tokenize
import jieba 

# Count_Words idea is to approximate the number of tokens in the sentence. We are assuming 1 word ~ 1.2 Tokens
def count_words(input_string,language):
    if language=='en':
        words = input_string.split()
        return round(1.2*len(words))
    else:
        return round(1.2*len(input_string))

# Function to add IDs to each Dataframe Row
def add_ids(row):
    global current_id
    # Add ID to the chunk
    row['Chunk'] = f'ID {current_id}: {row["Chunk"]}'
    current_id += 1
    return row

system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""

system_prompt_zh = """你将被赋予一个英文文档，其中段落用'ID XXXX: <text>'标识。

任务：找出第一个段落（不是第一个），其中内容与前面的段落相比较，显着地发生变化。

输出：以如下格式返回段落的ID：'Answer: ID XXXX'。

其他注意事项：避免形成过长的段落集群。考虑句子逻辑和语义内容。寻找一个平衡，以识别内容变化并保持段落集群长度适中。"""

def LLM_prompt(user_prompt,api_name,api_configure):
     match api_name:
        case 'zhipuai':
            try:
                from zhipuai import ZhipuAI
                client = ZhipuAI(api_key=api_configure['api_key'])
                response_glm = client.chat.completions.create(
                        model=api_configure['model_name'],  
                        messages= [
                                {"role": "system", "content": "You are an AI human assistant, answering user questions as perfectly as possible."},
                                {"role": "user", "content": user_prompt},
                            ],
                    )
                ans_glm=response_glm.choices[0].message.content
                # print(ans_glm)
                return ans_glm
            except Exception as e:
                if str(e) == "list index out of range":
                    print("LLM thinks prompt is unsafe")
                    return "content_flag_increment"
        case 'deepseek':
            raise ValueError("This model has not yet been implemented.")
        case _:
            raise ValueError("This model has not yet been implemented.")

def split_text_by_punctuation(text,language): 
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
        
        return sentences
    else:
        full_segments = sent_tokenize(text)
        ret = []
        for item in full_segments:
            item_l = item.strip().split(' ')
            if len(item_l) > 512:
                if len(item_l) > 1024:
                    item = ' '.join(item_l[:256]) + "..."
                else:
                    item = ' '.join(item_l[:512]) + "..."
            ret.append(item)
        return ret


def lumberchunker(api_name,api_configure,language,text,dynamic_merge='no',target_size=200):
    start_time = time.time() 

    id_sentence_list = split_text_by_punctuation(text,language)
    id_chunks = pd.DataFrame(id_sentence_list, columns=['Chunk'])  

    # Initialize a global variable for current_id and Apply the function along the rows of the DataFrame
    global current_id
    current_id = 0
    id_chunks = id_chunks.apply(add_ids, axis=1) # Put ID: Prefix before each paragraph

    chunk_number = 0
    new_id_list = []
    word_count_aux = []
    # print('len(id_chunks)',len(id_chunks))
    while chunk_number < len(id_chunks)-5:
        word_count = 0
        i = 0             
        while word_count < 550 and i+chunk_number<len(id_chunks)-1:
            i += 1
            final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
            word_count = count_words(final_document,'en')   
        

        if(i ==1):
            final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
        else:
            final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i-1 + chunk_number))
        
        
        question = f"\nDocument:\n{final_document}"

        word_count = count_words(final_document,'en')    
        word_count_aux.append(word_count)
        chunk_number = chunk_number + i-1
    
        if language=='en':
            prompt = system_prompt + question
        else:
            prompt = system_prompt_zh + question
        # print('111',prompt)
        gpt_output = LLM_prompt( prompt,api_name,api_configure)
        # print('222',gpt_output, flush=True)

        # For books where there is dubious content, Gemini refuses to run the prompt and returns mistake. This is to avoid being stalled here forever.
        if gpt_output == "content_flag_increment":
            chunk_number = chunk_number + 1

        else:
            pattern = r"Answer: ID \w+"
            match = re.search(pattern, gpt_output)

            if match == None:
                print("repeat this one")
            else:
                gpt_output1 = match.group(0)
                print(gpt_output1, flush=True)
                pattern = r'\d+'
                match = re.search(pattern, gpt_output1)
                chunk_number = int(match.group())
                new_id_list.append(chunk_number)
                if(new_id_list[-1] == chunk_number):
                    chunk_number = chunk_number + 1

    #Add the last chunk to the list
    new_id_list.append(len(id_chunks))

    # Remove IDs as they no longer make sense here.
    id_chunks['Chunk'] = id_chunks['Chunk'].str.replace(r'^ID \d+:\s*', '', regex=True)

    #Create final dataframe from chunks
    new_final_chunks = []
    # chapter_chunk = []
    for i in range(len(new_id_list)):
        # Calculate the start and end indices of each chunk
        start_idx = new_id_list[i-1] if i > 0 else 0
        end_idx = new_id_list[i]
        new_final_chunks.append(' '.join(id_chunks.iloc[start_idx: end_idx, 0]))
    
    if dynamic_merge!='no':
        merged_paragraphs = []  
        current_paragraph = "" 
        if language=='en':
            for paragraph in new_final_chunks:  
                # Check if adding a new paragraph to the current paragraph exceeds the target size
                if len(current_paragraph.split()) + len(paragraph.split()) <= target_size:  
                    current_paragraph +=' '+paragraph  
                else:  
                    merged_paragraphs.append(current_paragraph)  
                    current_paragraph = paragraph  
            if current_paragraph:  
                merged_paragraphs.append(current_paragraph)  
        else:
            for paragraph in new_final_chunks:  
                if len(current_paragraph) + len(paragraph) <= target_size:  
                    current_paragraph +=paragraph  
                else:  
                    merged_paragraphs.append(current_paragraph)  
                    current_paragraph = paragraph 
            if current_paragraph:  
                merged_paragraphs.append(current_paragraph) 
    else:
        merged_paragraphs = new_final_chunks   
          
    end_time = time.time()  
    execution_time = end_time - start_time  
    print(f"The program execution time is: {execution_time} seconds.")
    
    return merged_paragraphs