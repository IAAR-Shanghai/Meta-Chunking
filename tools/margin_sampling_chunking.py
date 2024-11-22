import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import time  
import jieba 

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

def get_prob_subtract(model,tokenizer,sentence1,sentence2,language):
    if language=='zh':
        query='''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
        1. 将“{}”分割成“{}”与“{}”两部分；
        2. 将“{}”不进行分割，保持原形式；
        请回答1或2。'''.format(sentence1+sentence2,sentence1,sentence2,sentence1+sentence2)
        prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids=prompt_ids
        output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()      
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()  
        prob_subtract=next_token_prob_1-next_token_prob_0
    else:
        query='''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
        1. Split "{}" into "{}" and "{}" two parts;
        2. Keep "{}" unsplit in its original form;
        Please answer 1 or 2.'''.format(sentence1+' '+sentence2,sentence1,sentence2,sentence1+' '+sentence2)
        prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids=prompt_ids
        output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()      
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()  
        prob_subtract=next_token_prob_1-next_token_prob_0
    return prob_subtract


def llm_chunker_ms(text,model,tokenizer,language,dynamic_merge,target_size):
    start_time = time.time() 
    save_list=[]

    threshold=0
    threshold_list=[]
    full_segments = split_text_by_punctuation(text,language)
    tmp=''
    for sentence in full_segments:
        if tmp=='':
            tmp+=sentence
        else:
            prob_subtract=get_prob_subtract(model,tokenizer,tmp,sentence,language)    
            threshold_list.append(prob_subtract)
            # print('222',prob_subtract)
            if prob_subtract>threshold:
                if language=='en':
                    tmp+=' '+sentence
                else:
                    tmp+=sentence
            else:
                save_list.append(tmp)
                tmp=sentence
        if len(threshold_list)>=5:
            last_ten = threshold_list[-5:]  
            avg = sum(last_ten) / len(last_ten)
            threshold=avg
    if tmp!='':
        save_list.append(tmp)
    
    new_final_chunks=save_list
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
    # 计算并打印执行时间  
    execution_time = end_time - start_time  
    print(f"The program execution time is: {execution_time} seconds.")
    
    return merged_paragraphs









