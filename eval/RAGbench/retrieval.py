import argparse
from loguru import logger
from base import BaseRetriever
from embeddings.base import HuggingfaceEmbeddings
import json
import pandas as pd 
from transformers import AutoModelForCausalLM, AutoTokenizer
from llms.base import BaseLLM

class Qwen_7B_Chat(BaseLLM):
    def __init__(self, model_name='qwen_7b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = 'Qwen2-7B-Instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto",
                                                     trust_remote_code=True).eval()
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }

    def request(self, query: str) -> str:
        query = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        input_ids = self.tokenizer.encode(query, return_tensors="pt").cuda()
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response

prefix = "Below is a question followed by some context from different sources. Please answer the question based on the context."

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='data/CUAD/test-00000-of-00001.parquet', help="Path to the dataset")  
parser.add_argument('--save_file', default='eval_result_dynamic/CUAD_baichuan_nodie_dynamic_00.json', help="Path to the answer")

parser.add_argument('--embedding_name', default='/data_extend/zjh/network/MultiHop-RAG-main/BAAI/bge-large-en-v1.5')
parser.add_argument('--embedding_dim', type=int, default=1024)

parser.add_argument('--docs_path', default='chunk_dynamic/CUAD_baichuan_nodie_dynamic_00.json', help="Path to the retrieval documents")  
parser.add_argument('--construct_index', action='store_true', help="Whether to construct an index")
parser.add_argument('--add_index', action='store_true', default=False, help="Whether to add an index")
parser.add_argument('--collection_name', default="CUAD_baichuan_nodie_dynamic_00", help="Name of the collection")   
parser.add_argument('--retrieve_top_k', type=int, default=5, help="Top k documents to retrieve")

args = parser.parse_args()
logger.info(args)

llm = Qwen_7B_Chat(model_name='qwen_7b', temperature=0.1, max_new_tokens=1280)
embed_model = HuggingfaceEmbeddings(model_name=args.embedding_name)
print('Finish Loading...')
retriever = BaseRetriever(
        args.docs_path, embed_model=embed_model, embed_dim=args.embedding_dim,
        construct_index=args.construct_index, add_index=args.add_index,
        collection_name=args.collection_name, similarity_top_k=args.retrieve_top_k
    )

print('Finish Indexing...')
df = pd.read_parquet(args.data_path)  
print("start to retrieve...")
retrieval_save_list = []
for index, row in df.iterrows(): 
    try:
        print(index,row['question'])
        retrieval_prompt=retriever.search_docs(row['question'])
        
        llm_ans=llm.request(retrieval_prompt)
        
        save = {}
        save['id'] = row['id']
        save['question'] = row['question']   
        save['response'] = row['response']   
        save['llm_ans'] = llm_ans
        save['retrieval_list'] = retrieval_prompt
        # save['gold_list'] =  gold_list  
        retrieval_save_list.append(save)
        # if index==1:
        #     break
    except:
        pass

with open(args.save_file, 'w') as json_file:
    json.dump(retrieval_save_list, json_file,indent=4)
    
# CUDA_VISIBLE_DEVICES=0,1 nohup python retrieval.py --construct_index >> eval_result_dynamic/CUAD_baichuan_nodie_dynamic_00_top5.log 2>&1 &
