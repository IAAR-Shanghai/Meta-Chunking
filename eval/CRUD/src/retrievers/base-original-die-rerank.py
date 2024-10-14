from abc import ABC

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.node_parser import SimpleNodeParser
from llama_index import download_loader

from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings
from llama_index.vector_stores import MilvusVectorStore
import os
from llama_index.data_structs import Node
import torch

def get_ppl_for_next(model,tokenizer,first_sentence,next_sentence):
    tokenized_text_1 = tokenizer(first_sentence, return_tensors="pt", add_special_tokens=False)
    tokenized_text_2 = tokenizer(next_sentence, return_tensors="pt", add_special_tokens=False)
    input_ids=torch.cat([tokenized_text_1["input_ids"].to(model.device),tokenized_text_2["input_ids"].to(model.device)],dim=-1)
    attention_mask = torch.cat([tokenized_text_1["attention_mask"].to(model.device),tokenized_text_2["attention_mask"].to(model.device)],dim=-1)
    with torch.no_grad():
        response = model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
    past_length=tokenized_text_1["input_ids"].to(model.device).shape[1]
    shift_logits = response.logits[..., past_length-1:-1, :].contiguous()  
    shift_labels = input_ids[..., past_length : ].contiguous()  
    active = (attention_mask[:, past_length:] == 1).view(-1)
    active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
    active_labels = shift_labels.view(-1)[active]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(active_logits, active_labels)  
    res = loss.mean().item()
    return res


class BaseRetriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            embed_dim: int = 768,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs",
            construct_index: bool = False,
            add_index: bool = False,
            similarity_top_k: int=2,
            llm=None
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self.llm = llm

        if construct_index:
            self.construct_index()
        else:
            self.load_index_from_milvus()
        
        if add_index:
            self.add_index()

        # self.query_engine = self.vector_index.as_query_engine()
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.similarity_top_k*2+8,
        )

        # assemble query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

    def construct_index(self):
        folder_path = self.docs_directory  
        nodes=[]
        chunk_size=218 
        for root, dirs, files in os.walk(folder_path):  
            for file in files:  
                relative_path = os.path.join(root, file)
                print(relative_path)
                with open(relative_path, 'r', encoding='utf-8') as file:  
                    content = file.read()  
                chunks = [] 
                chunks.append(content[0:chunk_size])
                i=chunk_size-50
                while i<len(content):
                    chunks.append(content[i:i+chunk_size])
                    i=i+chunk_size-50
                for i in chunks:
                    if len(i)<10:
                        continue
                    node1 = Node(text=i)
                    nodes.append(node1)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = MilvusVectorStore(
            dim=self.embed_dim, overwrite=True,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Process and index nodes in chunks due to Milvus limitations
        for spilt_ids in range(0, len(nodes), 8000):  
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], service_context=service_context, 
                storage_context=storage_context, show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")

            vector_store = MilvusVectorStore(
                overwrite=False,
                collection_name=self.collection_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("Indexing finished!")

    def add_index(self):
        if self.docs_type == 'json':
            JSONReader = download_loader("JSONReader")
            documents = JSONReader().load_data(self.docs_directory)
        else:
            documents = SimpleDirectoryReader(self.docs_directory).load_data()
        
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = MilvusVectorStore(
            overwrite=False,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

         # Process and index nodes in chunks due to Milvus limitations
        for spilt_ids in range(0, len(nodes), 8000):  
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], service_context=service_context, 
                storage_context=storage_context, show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")

        print("Indexing finished!")

    def load_index_from_milvus(self):
        vector_store =  MilvusVectorStore(
            overwrite=False, dim=self.embed_dim, 
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=None)
        self.vector_index = GPTVectorStoreIndex(
            [], storage_context=storage_context, 
            service_context=service_context,
        )

    def search_docs(self, query_text: str):
        response_vector = self.query_engine.query(query_text)

        response_text_list = response_vector.response.split('\n---------------------\n')
        response_text = response_text_list[1].split("\n\n")
        # response_text = "\n\n".join([text for text in response_text if not text.startswith("file_path: ")])
        raw_paragraphs=[text for text in response_text if not text.startswith("file_path: ")]
        print('raw_paragraphs',len(raw_paragraphs),raw_paragraphs)
        gen_question=query_text
        rrf_rank_ppl=[]
        rrf_rank_final={}
        for index, paragraph in enumerate(raw_paragraphs):
            gen_question_ppl=get_ppl_for_next(self.llm.model,self.llm.tokenizer,paragraph+'\n基于上述内容生成一个问题：',gen_question)
            rrf_rank_ppl.append(gen_question_ppl)
 
        sorted_lst = sorted(enumerate(rrf_rank_ppl), key=lambda x: x[1])   
        rank_dict = {value: idx + 1 for idx, (original_index, value) in enumerate(sorted_lst)}  
        for i in range(len(raw_paragraphs)):
            rrf_rank_final[i]={'sentence':raw_paragraphs[i],'similarity_ppl':[i+1,rank_dict[rrf_rank_ppl[i]]]}
        rrf_k=60
        for i,j in rrf_rank_final.items():
            j['rrf']=1.0 / (rrf_k+j['similarity_ppl'][0])+1.0 / (rrf_k+j['similarity_ppl'][1])
        rrf_rank_final = sorted(rrf_rank_final.items(), key=lambda x: x[1]['rrf'], reverse=True) 
        
        rerank_result=[]
        for i in range(self.similarity_top_k):
            rerank_result.append(rrf_rank_final[i][1]['sentence'])
        
        print('rerank_result',len(rerank_result),rerank_result)
        response_text = '\n\n'.join(rerank_result)
        
        return response_text

