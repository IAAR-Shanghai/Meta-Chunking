import math
from collections import defaultdict

# 整理数据，去除重复项
def remove_duplicates(edges):
    # 使用集合来记录已经遇到的节点和关系类型
    original_list = []
    for row_key, row_value in edges.items():
        for col_key, value in row_value.items():
            original_list.append(value['shortest_path'])

    seen = set()
    # 用于存储结果的列表
    unique_edges = []
             
    for sublist in original_list:
        for a, b in zip(sublist, sublist[1:]):
            key=(a,b)
            if (key not in seen):
                unique_edges.append({'row': a, 'column': b})
                seen.add(key)

    return unique_edges

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


import json
from perplexity_tools import *

# CUDA_VISIBLE_DEVICES=2 nohup python relation_eval2.py >> test_metric/relation_data_eval4/log_dijkstra/G3_chunk_semantic.log 2>&1 &

data_path='relation_data_eval1/G3_chunk_semantic.jsonl'
structural_entropy_val1=[]
structural_entropy_val2=[]
with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        Graph_3 = json.loads(line)
        # 创建一个新的字典，将字符串键转换为整数键
        converted_dict = {}
        for outer_key_str, inner_dict in Graph_3.items():
            outer_key = int(outer_key_str)  # 将外层的字符串键转换为整数
            converted_inner_dict = {}
            for inner_key_str, value in inner_dict.items():
                inner_key = int(inner_key_str)  # 将内层的字符串键也转换为整数
                converted_inner_dict[inner_key] = value
            converted_dict[outer_key] = converted_inner_dict
        Graph_3=converted_dict
        
        
        distance_path_3={}
        for start,_ in Graph_3.items():
            distance_path_3[start]=dijkstra_3(Graph_3, start)
        find_values=remove_duplicates(distance_path_3)
        graph_ent=build_graph(find_values)
        structural_entropy_val=calculate_structural_entropy(graph_ent)
        structural_entropy_val1.append(structural_entropy_val)
        
        
        # 用不完全图
        G_sentences=Graph_3
        Graph_3=create_graph_2(G_sentences,Graph_3)
        
        distance_path_3={}
        for start,_ in Graph_3.items():
            distance_path_3[start]=dijkstra_3(Graph_3, start)
        find_values=remove_duplicates(distance_path_3)
        graph_ent=build_graph(find_values)
        structural_entropy_val=calculate_structural_entropy(graph_ent)
        structural_entropy_val2.append(structural_entropy_val)
        
        
        avg_structural_entropy_val1=sum(structural_entropy_val1)/len(structural_entropy_val1)
        avg_structural_entropy_val2=sum(structural_entropy_val2)/len(structural_entropy_val2)
        print(avg_structural_entropy_val1,avg_structural_entropy_val2,flush=True)

avg_structural_entropy_val1=sum(structural_entropy_val1)/len(structural_entropy_val1)
avg_structural_entropy_val2=sum(structural_entropy_val2)/len(structural_entropy_val2)
print('end: ',avg_structural_entropy_val1,avg_structural_entropy_val2)
    
    
