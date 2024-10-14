from perplexity_chunking import Chunking
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import jieba 
import torch

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

def find_minima(values,threshold):  
        minima_indices = []  
        for i in range(1, len(values) - 1):  
            if values[i] < values[i - 1] and values[i] < values[i + 1]:
                if (values[i - 1]-values[i]>=threshold) or (values[i + 1]-values[i]>=threshold):
                    minima_indices.append(i)  
            elif values[i] < values[i - 1] and values[i] == values[i + 1]:
                if values[i - 1]-values[i]>=threshold:
                    minima_indices.append(i) 
        return minima_indices

def extract_by_html2text_db_nolist(sub_text,model,tokenizer,threshold,language='zh') -> List[str]:   
    temp_para=sub_text
    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para
    # 应用文本分割器  
    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)

    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
            index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices=find_minima(first_cluster_ppl,threshold)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    return final_chunks
    
