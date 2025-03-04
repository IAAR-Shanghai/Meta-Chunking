import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name_or_path= 'model/internlm3-8b-instruct' 
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,device_map=device_map)
model.eval()

##-------------------------------------
import math
from collections import defaultdict
# 构建图的节点度分布
def build_graph(edges):
    zgraph = defaultdict(int)  # 用于存储每个节点的度
    for edge in edges:
        node1 = edge['row']
        node2 = edge['column']
        zgraph[node1] += 1
        zgraph[node2] += 1
    return zgraph

# 计算结构熵
def calculate_structural_entropy(graph):
    total_degree = sum(graph.values())  # 所有节点的度总和
    entropy = 0
    for node, degree in graph.items():
        if degree > 0:
            p = degree / total_degree  # 节点的概率分布
            entropy -= p * math.log(p, 2)  # 信息熵公式
    
    return entropy

# 计算矩阵中所有值的平均
def calculate_average(matrix):
    total_sum = 0
    count = 0
    
    for row in matrix.values():
        for value in row.values():
            total_sum += value
            count += 1
            
    return total_sum / count

# 找到大于0.8的所有值和对应的键
def find_values_greater_than(matrix, threshold=0.8):
    results = []
    for row_key, row_value in matrix.items():
        for col_key, value in row_value.items():
            if value > threshold and row_key!=col_key:
                results.append({'row': row_key, 'column': col_key, 'value': value})
    return results


##-------------------------------------
from perplexity_tools import *
import json
data_path='relation_data/chunk_original.json'
with open(data_path, 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)
structural_entropy_val1=[]
structural_entropy_val2=[]
for item in  qa_data:
    segments = item['final_chunks']
    ch=Chunking(model, tokenizer)
    G_sentences=segments

    delta=0
    
    Graph_1=create_graph_1(G_sentences,ch)
    G_sentence_token_num=[tokenizer.encode(i, return_tensors='pt').to('cuda').shape[1] for i in G_sentences]
    Graph_3=create_graph_3(G_sentences,Graph_1,delta,G_sentence_token_num)
    find_values=find_values_greater_than(Graph_3, 0.8)
    graph_ent=build_graph(find_values)
    structural_entropy_val=calculate_structural_entropy(graph_ent)
    structural_entropy_val1.append(structural_entropy_val)
    # print(structural_entropy_val)

    Graph_2=create_graph_2(G_sentences,Graph_3)
    find_values=find_values_greater_than(Graph_2, 0.8)
    graph_ent=build_graph(find_values)
    structural_entropy_val=calculate_structural_entropy(graph_ent)
    structural_entropy_val2.append(structural_entropy_val)
    # print(structural_entropy_val)
    
    filename_z1 = 'relation_data_eval1/'+'G1_chunk_original.jsonl' 
    filename_z2 = 'relation_data_eval1/'+'G3_chunk_original.jsonl'  
    with open(filename_z1, 'a',encoding='utf-8') as f:  
        json.dump(Graph_1, f,ensure_ascii=False)  
        f.write('\n') 
    with open(filename_z2, 'a',encoding='utf-8') as f:  
        json.dump(Graph_3, f,ensure_ascii=False)  
        f.write('\n') 
    
    avg_structural_entropy_val1=sum(structural_entropy_val1)/len(structural_entropy_val1)
    avg_structural_entropy_val2=sum(structural_entropy_val2)/len(structural_entropy_val2)
    print('end: ',avg_structural_entropy_val1,avg_structural_entropy_val2,flush=True)

avg_structural_entropy_val1=sum(structural_entropy_val1)/len(structural_entropy_val1)
avg_structural_entropy_val2=sum(structural_entropy_val2)/len(structural_entropy_val2)
print('end: ',avg_structural_entropy_val1,avg_structural_entropy_val2)


# CUDA_VISIBLE_DEVICES=3 nohup python relation_eval.py >> test_metric/relation_data_eval4/chunk_original.log 2>&1 &