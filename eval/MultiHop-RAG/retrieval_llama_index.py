import argparse
from loguru import logger
from base_llama_index import BaseRetriever
from embeddings.base import HuggingfaceEmbeddings
import json


parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='dataset/MultiHopRAG.json', help="Path to the dataset")
parser.add_argument('--save_file', default='toy_data/docs_llama_index_corpus_nodie_top10.json', help="Path to the answer")

parser.add_argument('--embedding_name', default='BAAI/bge-large-en-v1.5')
parser.add_argument('--embedding_dim', type=int, default=1024)

parser.add_argument('--docs_path', default='dataset/original', help="Path to the retrieval documents")
parser.add_argument('--construct_index', action='store_true', help="Whether to construct an index")
parser.add_argument('--add_index', action='store_true', default=False, help="Whether to add an index")
parser.add_argument('--collection_name', default="docs_llama_index_corpus_nodie", help="Name of the collection")
parser.add_argument('--retrieve_top_k', type=int, default=10, help="Top k documents to retrieve")

args = parser.parse_args()
logger.info(args)


embed_model = HuggingfaceEmbeddings(model_name=args.embedding_name)
print('Finish Loading...')
retriever = BaseRetriever(
        args.docs_path, embed_model=embed_model, embed_dim=args.embedding_dim,
        construct_index=args.construct_index, add_index=args.add_index,
        collection_name=args.collection_name, similarity_top_k=args.retrieve_top_k
    )
print('Finish Indexing...')
with open(args.data_path, 'r', encoding='utf-8') as file:  
    multihop_data = json.load(file)
retrieval_save_list = []
print("start to retrieve...")
i=0
for query_data in multihop_data:
    response_list=retriever.search_docs(query_data['query'])
    retrieval_list=[]
    for sentence in response_list:
        tmp={}
        tmp["text"]=sentence
        retrieval_list.append(tmp)
    
    save = {}
    save['query'] = query_data['query']   
    save['answer'] = query_data['answer']   
    save['question_type'] = query_data['question_type'] 
    save['retrieval_list'] = retrieval_list
    save['gold_list'] = query_data['evidence_list']   
    retrieval_save_list.append(save)
    print('order: ',str(i),' response_list: ',len(response_list))
    i+=1
with open(args.save_file, 'w') as json_file:
    json.dump(retrieval_save_list, json_file,indent=4)
    
# CUDA_VISIBLE_DEVICES=5 nohup python retrieval_llama_index.py --construct_index >> docs_llama_index_corpus_nodie_top10.log 2>&1 &