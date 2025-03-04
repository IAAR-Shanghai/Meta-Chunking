import math 
import heapq  
import copy
from perplexity_chunking import *  



def create_graph_1(G_sentences,ch):
    G_sentences_ppl=[]  #实际上这个在上面初步切分时已经算出来了
    for sentence in G_sentences:
        # sentence, tokens, probabilities = ch.generate_tokens_and_prob(' ', 0, sentence) 
        # weight=ch.get_sen_log_prob(probabilities)
        sum_ppl, _, ppl_len = ch.get_ppl_for_next(' ',sentence)
        G_sentences_ppl.append(sum_ppl)

    graph={}
    for i in range(len(G_sentences)):  
        graph[i] = {}
    for i in range(len(G_sentences)):  
        for j in range(len(G_sentences)):    
            if i==j:
                graph[i][i]=G_sentences_ppl[i]
            else:
                # sentence, tokens, probabilities = ch.generate_tokens_and_prob(G_sentences[i], 0, G_sentences[j]) 
                # weight=ch.get_sen_log_prob(probabilities)
                sum_ppl, _, ppl_len = ch.get_ppl_for_next(G_sentences[i],  G_sentences[j])
                graph[i][j]=sum_ppl
    return graph

def create_graph_2(G_sentences,Graph):
    graph={}
    for i in range(len(G_sentences)):  
        graph[i] = {}
    for i in range(len(G_sentences)):  
        for j in range(i,len(G_sentences)):  # 只添加一半边，因为图是全连接的  
            if i==j:
                graph[i][i]=Graph[i][i]
            else:
                graph[i][j]=Graph[i][j]
    return graph

def create_graph_3(G_sentences,Graph,delta,G_sentence_token_num):
    graph={}
    for i in range(len(G_sentences)):  
        graph[i] = {}
    for i in range(len(G_sentences)):  
        for j in range(len(G_sentences)):    
            if i==j:
                graph[i][i]=1
            else:
                weight_temp=(math.exp(Graph[j][j] /G_sentence_token_num[j])-math.exp(Graph[i][j] /G_sentence_token_num[j]))/math.exp(Graph[j][j] /G_sentence_token_num[j])   #注意一个句子时不需要分组
                weight=-weight_temp+1+delta*abs(i-j)/(len(G_sentences)-1)
                graph[i][j]=weight
    return graph

def has_cycle(edges_dict):  #深度优先搜索判断有没有死循环
    visited = set()  
    recursion_stack = set()  
      
    def dfs(node):  
        if node in recursion_stack:  
            return True  
        if node in visited:  
            return False  
        visited.add(node)  
        recursion_stack.add(node)  
        neighbor = edges_dict.get(node)  
        if neighbor is not None and dfs(neighbor):  
            return True  
        recursion_stack.remove(node)  
        return False  
      
    for node in edges_dict.keys():  
        if dfs(node):  
            return True  
    return False 

def dijkstra_1(graph, start,G_sentence_token_num):  
    # 初始化距离字典，将所有节点的距离设置为无穷大，除了起始节点设置为0  
    distances = {node: float('infinity') for node in graph}  
    distances[start] = graph[start][start]/G_sentence_token_num[start] 
    # 计数到每个节点的最短路径的token数量
    numbers = {node: 0 for node in graph}  
    numbers[start] =G_sentence_token_num[start] 

    # 使用优先队列（最小堆）来存储待处理的节点  
    pq = [(distances[start], start)]  

    # 记录路径  
    path = {node: None for node in graph}  
    path_temp= copy.deepcopy(path) 

    while pq:   
        current_distance, current_node = heapq.heappop(pq)  
        # 如果当前节点的距离已经被更新过（即不是最小的），则跳过  
        if current_distance > distances[current_node]:  
            continue
        # 遍历当前节点的邻居  
        for neighbor, weight in graph[current_node].items(): 
            if neighbor != current_node:
                distance = (current_distance*numbers[current_node] + weight ) /(numbers[current_node]+G_sentence_token_num[neighbor])
                # print(distance ,distances[neighbor])

                # 如果找到了更短的路径，则更新距离和路径  
                if distance < distances[neighbor]:  
                    path_temp[neighbor] = current_node
                    if not has_cycle(path_temp):
                        distances[neighbor] = distance 
                        numbers[neighbor] = numbers[current_node]+G_sentence_token_num[neighbor]
                        path[neighbor] = current_node  
                        heapq.heappush(pq, (distance, neighbor))  
                    else:
                        path_temp= copy.deepcopy(path) 
                        continue
    # 构建最短路径
    result={}
    for znode,zdistance in distances.items():
        if znode==start:
            result[znode]={'distance':graph[start][start]/G_sentence_token_num[start] ,'shortest_path':[znode]}
            continue
        shortest_path = [] 
        end_path=znode 
        while end_path is not None:  
            shortest_path.append(end_path)  
            end_path = path[end_path]  
        shortest_path.reverse()  
        result[znode]={'distance':zdistance,'shortest_path':shortest_path}

    return result

def dijkstra_2(graph, start,G_sentence_token_num):  
    # 初始化距离字典，将所有节点的距离设置为无穷大，除了起始节点设置为0  
    distances = {node: float('infinity') for node in graph}  
    distances[start] = graph[start][start]/G_sentence_token_num[start] 
    # 计数到每个节点的最短路径的token数量
    numbers = {node: 0 for node in graph}  
    numbers[start] =G_sentence_token_num[start] 

    # 使用优先队列（最小堆）来存储待处理的节点  
    pq = [(distances[start], start)]  

    # 记录路径  
    path = {node: None for node in graph}  

    while pq:   
        current_distance, current_node = heapq.heappop(pq)  
        # 如果当前节点的距离已经被更新过（即不是最小的），则跳过  
        if current_distance > distances[current_node]:  
            continue
        # 遍历当前节点的邻居  
        for neighbor, weight in graph[current_node].items(): 
            if neighbor != current_node:
                distance = (current_distance*numbers[current_node] + weight ) /(numbers[current_node]+G_sentence_token_num[neighbor])
                # print(distance ,distances[neighbor])

                # 如果找到了更短的路径，则更新距离和路径  
                if distance < distances[neighbor]:  
                    distances[neighbor] = distance 
                    numbers[neighbor] = numbers[current_node]+G_sentence_token_num[neighbor]
                    path[neighbor] = current_node  
                    heapq.heappush(pq, (distance, neighbor))  
        

    # 构建最短路径
    result={}
    for znode,zdistance in distances.items():
        if znode<start:
            continue
        if znode==start:
            result[znode]={'distance':graph[start][start]/G_sentence_token_num[start] ,'shortest_path':[znode]}
            continue
        shortest_path = [] 
        end_path=znode 
        while end_path is not None:  
            shortest_path.append(end_path)  
            end_path = path[end_path]  
        shortest_path.reverse()  
        result[znode]={'distance':zdistance,'shortest_path':shortest_path}

    return result

def dijkstra_3(graph, start):  
    # 初始化距离字典，将所有节点的距离设置为无穷大，除了起始节点设置为0  
    distances = {node: float('infinity') for node in graph}  
    distances[start] = 0
    # 计数到每个节点的最短路径的token数量
    numbers = {node: 0 for node in graph}  

    # 使用优先队列（最小堆）来存储待处理的节点  
    pq = [(0, start)]  

    # 记录路径  
    path = {node: None for node in graph}  
    path_temp= copy.deepcopy(path) 

    while pq:   
        current_distance, current_node = heapq.heappop(pq)  
        # 如果当前节点的距离已经被更新过（即不是最小的），则跳过  
        if current_distance > distances[current_node]:  
            continue
        # 遍历当前节点的邻居  
        for neighbor, weight in graph[current_node].items(): 
            if neighbor != current_node:
                distance = (current_distance*numbers[current_node]+weight)/(numbers[current_node]+1)
                # print(distance ,distances[neighbor])

                # 如果找到了更短的路径，则更新距离和路径  
                if distance < distances[neighbor]:  
                    path_temp[neighbor] = current_node
                    if not has_cycle(path_temp):
                        distances[neighbor] = distance 
                        numbers[neighbor] = numbers[current_node]+1
                        path[neighbor] = current_node  
                        heapq.heappush(pq, (distance, neighbor))  
                    else:
                        path_temp= copy.deepcopy(path) 
                        continue

        

    # 构建最短路径
    result={}
    for znode,zdistance in distances.items():
        if znode==start:
            # result[znode]={'distance':graph[start][start]/G_sentence_token_num[start] ,'shortest_path':[znode]}
            continue
        shortest_path = [] 
        end_path=znode 
        while end_path is not None:  
            shortest_path.append(end_path)  
            end_path = path[end_path]  
        shortest_path.reverse()  
        result[znode]={'distance':zdistance,'shortest_path':shortest_path}
    # print(numbers)

    return result


