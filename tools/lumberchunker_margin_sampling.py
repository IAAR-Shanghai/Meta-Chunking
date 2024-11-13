import time
import pandas as pd
import time  
from nltk.tokenize import sent_tokenize
import jieba 
import torch
import torch.nn.functional as F


# Count_Words idea is to approximate the number of tokens in the sentence. We are assuming 1 word ~ 1.2 Tokens
def count_words(input_string,language):
    if language=='en':
        words = input_string.split()
        return round(1.2*len(words))
    else:
        return round(1.2*len(input_string))

system_prompt = """You will receive a document as input, where each paragraph is identified by "<ID>: <text>".

Task: Find the first paragraph (not the first one) that shows a clear change in content compared to the previous paragraphs.

Output: Directly return the ID of the paragraph where the content changes.

Other notes: Avoid forming excessively long paragraph clusters. Comprehensively consider the logical structure and semantic content of sentences. Seek a good balance between identifying content changes and maintaining manageable paragraph clusters."""

system_prompt_zh = """您将会接收到一个文档作为输入，其中各段落通过“<ID>：<文本>”进行标识。

任务：找到第一个（非首个）与前一段落内容相比发生明显变化的段落。

输出：直接返回内容发生变化的段落的ID。

其他注意事项：避免形成过长的段落群。综合考虑句子的逻辑结构和语义内容。要在识别内容变化和保持段落群的可管理性之间寻求良好的平衡。"""

def get_maxprob_index(model,tokenizer,user_prompt,len_documents):
    prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(user_prompt)
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    input_ids=prompt_ids
    output_ids = [tokenizer.encode(str(i), return_tensors='pt').to(model.device)[0].cpu().tolist() for i in range(len_documents)]
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        token_probs = F.softmax(next_token_logits, dim=-1)
    prob_list=[]
    for ids in output_ids:
        next_token_id = ids[0] 
        next_token_prob = token_probs[:, next_token_id].item()  
        if len(ids)==1:
            prob_list.append(next_token_prob)
        else:
            toavg_prob=[next_token_prob]
            for id in ids[1:]:
                with torch.no_grad():
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)],dim=-1)
                    outputs = model(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    token_probs = F.softmax(next_token_logits, dim=-1)
                next_token_prob = token_probs[:, id].item()  
                toavg_prob.append(next_token_prob)
                next_token_id=id
            prob_list.append(sum(toavg_prob) / len(toavg_prob))
                
    # print(prob_list)
    max_value = max(prob_list)  
    # Find the index of the maximum value 
    max_prob_index = prob_list.index(max_value)  
    
    return max_prob_index

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


def lumberchunker_ms(small_tokenizer,small_model,language,text,dynamic_merge='no',target_size=200):
    start_time = time.time() 
    id_sentence_list=[]

    id_sentence_list = split_text_by_punctuation(text,language)

    id_chunks = pd.DataFrame(id_sentence_list, columns=['Chunk'])  

    chunk_number = 0

    new_final_chunks = []

    word_count_aux = []
    # print('len(id_chunks)',len(id_chunks))
    while chunk_number < len(id_chunks)-5:
        word_count = 0
        i = 0             
        while word_count < 550 and i+chunk_number<len(id_chunks)-1:
            i += 1
            final_document_str = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
            word_count = count_words(final_document_str,language)  

        if(i ==1):
            final_document = [id_chunks.at[k, 'Chunk'] for k in range(chunk_number, i + chunk_number)]
        else:
            final_document = [id_chunks.at[k, 'Chunk'] for k in range(chunk_number, i-1 + chunk_number)]
        
        len_documents=len(final_document)
        final_document_list2str=''
        for i,j in enumerate(final_document):
            final_document_list2str += '\n'+str(i)+': '+j
        question = f"\nDocument:{final_document_list2str}"

        word_count = count_words(question,language)    
        word_count_aux.append(word_count)
        if language=='en':
            prompt = system_prompt + question
        else:
            prompt = system_prompt_zh + question
        gpt_output = get_maxprob_index(small_model,small_tokenizer,prompt,len_documents)
        
        if gpt_output!=0:
            chunk_number = chunk_number + gpt_output
            new_final_chunks.append(' '.join([j for i,j in enumerate(final_document) if i<int(gpt_output)]))
        else:
            chunk_number = chunk_number + 1
            new_final_chunks.append(final_document[0])

    final_document = [id_chunks.at[k, 'Chunk'] for k in range(chunk_number, len(id_chunks))]
    new_final_chunks.append(' '.join(final_document))
    
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
    # Calculate and print execution time 
    execution_time = end_time - start_time  
    print(f"The program execution time is: {execution_time} seconds.")
    
    return merged_paragraphs

