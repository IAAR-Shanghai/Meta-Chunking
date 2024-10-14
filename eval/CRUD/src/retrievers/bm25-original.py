from abc import ABC

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import ElasticsearchStore

from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings
from llama_index import QueryBundle
from elasticsearch import Elasticsearch
import os
from llama_index.data_structs import Node

class CustomBM25Retriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs_80k",
            construct_index: bool = False,
            similarity_top_k: int=2,
            es_host: str = '127.0.0.1',
            es_port: int = 9221,
            es_scheme: str = 'http',
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.collection_name = collection_name
        self.es_host = es_host
        self.es_port = es_port
        self.es_scheme = es_scheme

        if construct_index:
            self.construct_index()

        self.es_client = Elasticsearch([{'host': self.es_host, 'port': self.es_port, "scheme": self.es_scheme}])
        print("Elasticsearch connected!")

    def construct_index(self):
        folder_path = self.docs_directory  
        nodes=[]
        chunk_size=218
        # 遍历文件夹  
        for root, dirs, files in os.walk(folder_path):  
            for file in files:  
                relative_path = os.path.join(root, file)
                print(relative_path)
                with open(relative_path, 'r', encoding='utf-8') as file:  
                    # 读取文件内容  
                    content = file.read()  
                # 打印文件内容  
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]  
                for i in chunks:
                    if len(i)<10:
                        continue
                    node1 = Node(text=i)
                    nodes.append(node1)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = ElasticsearchStore(
            index_name=self.collection_name, es_url=f"{self.es_scheme}://{self.es_host}:{self.es_port}"
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

    def search_docs(self, query_text: str):
        query = QueryBundle(query_text)

        result = []
        dsl = {
            'query': {
                'match': {
                    'content': query.query_str
                }
            },
            "size": self.similarity_top_k
        }
        search_result = self.es_client.search(index=self.collection_name, body=dsl)
        if search_result['hits']['hits']:
            for record in search_result['hits']['hits']:
                text = record['_source']['content']
                result.append(text)
        
        result_str = '\n'.join(result)

        return result_str

