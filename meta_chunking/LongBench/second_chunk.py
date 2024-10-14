from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

model_name_or_path="Qwen2-72B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True,device_map="auto").eval()
device=model.device

prompt = '''You are an expert in text segmentation, tasked with dividing given text into blocks. You must adhere to the following four conditions:
1. Segment the text based solely on its logical and semantic structures.
2. Aim to keep each block around 87 English words in length.
3. Do not alter the original vocabulary or structure of the text.
4. Do not add any new words or symbols.
By solely determining the boundaries for text segmentation, divide the original text into blocks and output them individually, separated by a clear delimiter '--- Block Separator ---'. Do not output any other explanations. If you understand, please proceed to segment the following text into blocks: '''

save_filename='data/llm_2wikimqa.json'
with open('data/2wikimqa.json', 'r', encoding='utf-8') as file:  
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
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    blocks = response.split("--- Block Separator ---")  
    llm_response.append(response)
    llm_blocks+=blocks
    with open('data/tmp_response.json', 'w') as file:
        json.dump(llm_response, file)
    # if i<5:
    #     i+=1
    # else:
    #     break
with open(save_filename, 'w') as file:
    json.dump(llm_blocks, file)
    
# CUDA_VISIBLE_DEVICES=1,2,3,4,5 nohup python second_chunk.py >> logs/llm_2wikimqa.log 2>&1 &