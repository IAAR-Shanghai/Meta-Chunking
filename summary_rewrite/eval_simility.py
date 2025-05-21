from src.embeddings.base import HuggingfaceEmbeddings
from sentence_transformers import SentenceTransformer
from src.eval_retrieval import BaseRetriever
import json


model_name_or_path='BAAI/bge-large-zh-v1.5'
model_sim = SentenceTransformer(model_name_or_path)
embed_model = HuggingfaceEmbeddings(model_name='/BAAI/bge-base-zh-v1.5')


def get_simscore_for_next(model,first_sentence,next_sentence):
    queries = [next_sentence]
    passages = [first_sentence]
    q_embeddings = model.encode([q for q in queries], normalize_embeddings=True)
    p_embeddings = model.encode(passages, normalize_embeddings=True)
    scores = q_embeddings @ p_embeddings.T
    return scores.item()

with open('data/crud_split/split_merged.json', 'r', encoding='utf-8') as cfile: 
    qa_data = json.load(cfile)
    
retriever = BaseRetriever(
        '', embed_model=embed_model, embed_dim=768,
        chunk_size=128, chunk_overlap=50,
        construct_index=False, add_index=False,
        collection_name='db_qa_baichuan_rewrite_yiyan_all', similarity_top_k=8
    )

# 需要创建索引
# retriever = BaseRetriever(
#         'add_shiyan/CRUD/db_qa_qwen7B_nodie_dynamic_00_merge.json', embed_model=embed_model, embed_dim=768,
#         chunk_size=128, chunk_overlap=50,
#         construct_index=True, add_index=False,
#         collection_name='db_qa_qwen7B_nodie_dynamic_00_merge', similarity_top_k=8
#     )
filename='eval_simility/db_qa_baichuan_rewrite_yiyan_all_topk8.json'

sum_allsim=0
all_num=0
all_data=[]
for item in qa_data['questanswer_1doc']:
    try:
        news1=item['news1']
        questions=item['questions']
        data=retriever.search_docs(questions)
        sim_texts = []
        for i in range(len(data)):
            sim_text = get_simscore_for_next(model_sim, news1, data[i])
            sim_texts.append(sim_text)
        # 先计算每个的平均值，再计算所有的平均值
        avg_sim = sum(sim_texts) / len(sim_texts)
        print('111',avg_sim,flush=True)
        save={
            "ID": item['ID'],
            "event": item['event'],
            "news1": news1,
            "questions": questions,
            "answers": item['answers'],
            "data": data,
            "sim_score": sim_texts
        }
        sum_allsim+=avg_sim
        all_num+=1
        all_data.append(save) 
        with open(filename, 'w', encoding='utf-8') as sfile:
            json.dump(all_data, sfile, ensure_ascii=False, indent=4)
    except:
        pass
all_avg_sim = sum_allsim / all_num
print('final_avg_sim: ',all_avg_sim,flush=True)

# CUDA_VISIBLE_DEVICES=6 nohup python eval_simility.py >> eval_simility/db_qa_baichuan_rewrite_yiyan_all_topk8.log 2>&1 &







