from metric.common import (
    bleu_score, 
    rougeL_score, 
    bert_score,
)
import json
from tqdm import tqdm


def scoring(generated_text,ground_truth_text) -> dict:
    bertscore = bert_score(generated_text, ground_truth_text)
    
    bleu_avg, bleu1, bleu2, bleu3, bleu4 = bleu_score(generated_text, ground_truth_text)

    return {
        'metrics': {
            'bleu-avg': bleu_avg or 0.0,
            'bleu-1': bleu1 or 0.0,
            'bleu-2': bleu2 or 0.0,
            'bleu-3': bleu3 or 0.0,
            'bleu-4': bleu4 or 0.0,
            'rouge-L': rougeL_score(generated_text, ground_truth_text) or 0.0,
            'bertScore': bertscore,
            'length': len(generated_text.split())
        },
        'valid': len(generated_text.split()) != 0
    }

file_path='eval_result_dynamic/CUAD_baichuan_nodie_dynamic_00.json'
save_file='eval_result_dynamic/CUAD_baichuan_nodie_dynamic_00_eval.json'
with open(file_path, 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)

eval_result=[]
for qa in tqdm(qa_data):
    generated_text=qa['llm_ans']
    ground_truth_text=qa['response']
    try:
        qa_sc=scoring(generated_text,ground_truth_text)
        eval_result.append(qa_sc)
    except:
        pass
            
with open(save_file, 'w') as json_file:
    json.dump(eval_result, json_file,indent=4)

bleu_avg=0
bleu_1=0
bleu_2=0 
bleu_3=0 
bleu_4=0      
rouge_L=0   
bertScore=0  
length=0
i=0
for rs in eval_result:
    if rs['valid']==True:
        bleu_avg+=rs['metrics']['bleu-avg']
        bleu_1+=rs['metrics']['bleu-1']
        bleu_2+=rs['metrics']['bleu-2']
        bleu_3+=rs['metrics']['bleu-3']
        bleu_4+=rs['metrics']['bleu-4']      
        rouge_L+=rs['metrics']['rouge-L']  
        bertScore+=rs['metrics']['bertScore']
        length+=rs['metrics']['length']
        i+=1
res={
    'bleu-avg': bleu_avg/i,
    'bleu-1': bleu_1/i,
    'bleu-2': bleu_2/i,
    'bleu-3': bleu_3/i,
    'bleu-4': bleu_4/i,
    'rouge-L': rouge_L/i,
    'bertScore': bertScore/i,
    'length': length/i
}
print(res)


# CUDA_VISIBLE_DEVICES=5 nohup python evaluate_qa.py >> eval_result_dynamic/CUAD_baichuan_nodie_dynamic_00_eval.log 2>&1 &