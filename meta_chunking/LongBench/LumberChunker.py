# Used for English datasets
import time
import re
import pandas as pd
from tqdm import tqdm
import argparse
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import time  
import json
from nltk.tokenize import sent_tokenize
import jieba 

model_name_or_path= 'Qwen2-7B-Instruct'   
device_map = "auto"
small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map) 
small_model.eval()

# Argument parsing
parser = argparse.ArgumentParser(description="Process some text.")
parser.add_argument('--out_path', type=str, default='LumberChunker', help='Output directory path')
args = parser.parse_args()
out_path = args.out_path
fileOut  = f'{out_path}/qasper_qwen2_7B_Chunks_310.json'

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# Ensure the output directory exists
create_directory(args.out_path)


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


def LLM_prompt(user_prompt):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
        text = small_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = small_tokenizer([text], return_tensors="pt").to(small_model.device)

        generated_ids = small_model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = small_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        if str(e) == "list index out of range":
            print("GPT thinks prompt is unsafe")
            return "content_flag_increment"

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

# dataset = pd.read_parquet("hf://datasets/LumberChunker/GutenQA_Paragraphs/GutenQA_paragraphs.parquet", engine="pyarrow")

# # Filter the DataFrame to show only rows with the specified book name
# paragraph_chunks = dataset[dataset['Chunk ID'] <10].reset_index(drop=True)

# # Check if the filtered DataFrame is empty
# if paragraph_chunks.empty:
#     sys.exit("Choose a valid book name!")

# id_chunks = paragraph_chunks['Chunk'].to_frame()
start_time = time.time() 
# i=0
id_chunks_list=[]
with open('data/qasper.jsonl', 'r', encoding='utf-8') as file:  

    for line in file:  

        data = json.loads(line)
        segments = split_text_by_punctuation(data['context'],'en')       
        id_chunks_list=id_chunks_list+segments
        # if i==10:
        #     break
        # else:
        #     i+=1
id_chunks = pd.DataFrame(id_chunks_list, columns=['Chunk'])  

# Initialize a global variable for current_id and Apply the function along the rows of the DataFrame
current_id = 0
id_chunks = id_chunks.apply(add_ids, axis=1) # Put ID: Prefix before each paragraph


chunk_number = 0
i = 0


new_id_list = []

word_count_aux = []
current_iteration = 0
print('len(id_chunks)',len(id_chunks))
while chunk_number < len(id_chunks)-5:
    word_count = 0
    i = 0             #550
    while word_count < 310 and i+chunk_number<len(id_chunks)-1:
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

    prompt = system_prompt + question
    # print('111',prompt)
    gpt_output = LLM_prompt( user_prompt=prompt)
    print('222',gpt_output, flush=True)


    # For books where there is dubious content, Gemini refuses to run the prompt and returns mistake. This is to avoid being stalled here forever.
    if gpt_output == "content_flag_increment":
        chunk_number = chunk_number + 1

    else:
        # pattern = r"Answer: ID \w+"
        pattern = r"ID \d+"
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
    new_final_chunks.append('\n'.join(id_chunks.iloc[start_idx: end_idx, 0]))
    

    # #When building final Dataframe, sometimes text from different chapters is concatenated. When this happens we update the Chapter column accordingly.
    # if(paragraph_chunks["Chapter"][start_idx] != paragraph_chunks["Chapter"][end_idx-1]):
    #     chapter_chunk.append(f"{paragraph_chunks['Chapter'][start_idx]} and {paragraph_chunks['Chapter'][end_idx-1]}")
    # else:
    #     chapter_chunk.append(paragraph_chunks['Chapter'][start_idx])

# # Write new Chunks Dataframe
# df_new_final_chunks = pd.DataFrame({'Chapter': chapter_chunk, 'Chunk': new_final_chunks})
# df_new_final_chunks.to_excel(fileOut, index=False)
# print("Completed!")
with open(fileOut, 'w') as file:
    json.dump(new_final_chunks, file)
    
end_time = time.time()  

execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")

# CUDA_VISIBLE_DEVICES=5 nohup python LumberChunker.py >> LumberChunker/qasper_qwen2_7B_Chunks_310.log 2>&1 &