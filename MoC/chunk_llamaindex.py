from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
import json
import os
import time


# llama_index切分
# root_dir = 'db_qa_tmp'
# all_data=[]
# for root, dirs, files in os.walk(root_dir):  
#     for file in files:
#         documents = SimpleDirectoryReader(input_files=[os.path.join(root, file)]).load_data() 
#         node_parser = SimpleNodeParser.from_defaults(
#             chunk_size=196, chunk_overlap=0)
#         nodes_tmp = node_parser.get_nodes_from_documents(documents, show_progress=True)

#         nodes=[]
#         for node in nodes_tmp:
#             i=node.text
#             nodes.append(i)
        
#         with open(os.path.join(root, file), 'r', encoding='utf-8') as cfile:  
#             content = cfile.read()
#         save = {}
#         save['raw_corpus'] = content
#         save['final_chunks'] = nodes
#         all_data.append(save)
            
# with open('chunk_llamaindex.json', 'w', encoding='utf-8') as sfile:
#     json.dump(all_data, sfile, ensure_ascii=False, indent=4)
            
# # nohup python chunk_llamaindex.py &


# # 原始切分
# root_dir = 'db_qa_tmp'
# all_data=[]
# for root, dirs, files in os.walk(root_dir):  
#     for file in files:
#         chunk_size=134
#         with open(os.path.join(root, file), 'r', encoding='utf-8') as file:  
#             # 读取文件内容  
#             content = file.read()  
#         # 打印文件内容  
#         chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)] 
        
#         save = {}
#         save['raw_corpus'] = content
#         save['final_chunks'] = chunks
#         all_data.append(save)
            
# with open('chunk_original.json', 'w', encoding='utf-8') as sfile:
#     json.dump(all_data, sfile, ensure_ascii=False, indent=4)
            
# # nohup python chunk_llamaindex.py &


# llama_index切分
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
import time

start_time = time.time() 
save_path='chunking/webcpm_oe/webcpm_semantic_bgelarge.json'
split_path='/users/u2023000898/moe_test/chunking/data/webcpm'
documents = SimpleDirectoryReader(split_path).load_data()

embed_model=HuggingFaceEmbedding(model_name="/users/u2023000898/model/BAAI/bge-large-zh-v1.5")  
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=74, embed_model=embed_model   
)                   # 汉语用自己的句子分割函数split_by_my_zh

all_data=[]
for item in documents:
    nodes = splitter.get_nodes_from_documents([item], show_progress=True)

    for node in nodes:
        if node.text.strip() !='':
            all_data.append(node.text)
        
with open(save_path, 'w', encoding='utf-8') as sfile:
    json.dump(all_data, sfile, ensure_ascii=False, indent=4)
end_time = time.time()  
# 计算并打印执行时间  
execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")

# CUDA_VISIBLE_DEVICES=1 nohup python chunk_llamaindex.py &