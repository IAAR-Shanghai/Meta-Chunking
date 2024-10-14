from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
import time

start_time = time.time() 
save_path='chunk_semantic/multifieldqa_zh_73.json'
split_path='tmp/multifieldqa_zh'
documents = SimpleDirectoryReader(split_path).load_data()

embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1") 
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=73, embed_model=embed_model
)
nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

save_list=[]
for node in nodes:
    save_list.append(node.text)
    
with open(save_path, 'w') as file:
    json.dump(save_list, file)
end_time = time.time()  

execution_time = end_time - start_time  
print(f"程序执行时间为: {execution_time} 秒")

# CUDA_VISIBLE_DEVICES=3 nohup python semantic_chunk.py >> chunk_semantic/multifieldqa_zh_73.log 2>&1 &


