import sys
from abc import ABC, abstractmethod
from llama_index.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.ingestion import IngestionPipeline
from llama_index.vector_stores import PGVectorStore
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.azurecosmosmongo import (
    AzureCosmosDBMongoDBVectorSearch,
)
import pymongo
import logging


# logger
logging.basicConfig(format='%(asctime)s %(clientip)-15s %(user)-8s %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


# classes for vector_store
class BaseVectorStore(ABC):
    def create_vector_embedding(self, nodes: object,
                                llm: object,
                                embed_model: object) -> object:
        '''
        To create vector embedding on nodes
        '''

        # extractors
        extractors = [
            TitleExtractor(nodes=5, llm=llm),
            QuestionsAnsweredExtractor(questions=3, llm=llm)
        ]

        # ingestion pipeline
        pipeline = IngestionPipeline(
            transformations=extractors,
        )
        nodes = pipeline.run(nodes=nodes, in_place=False)

        for index, node in enumerate(nodes):
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode='all')
            )
            node.embedding = node_embedding
        
        return nodes

    @abstractmethod
    def insert_vectors_from_nodes():
        '''
        To insert the documents to the vector store
        '''
        pass

    @abstractmethod                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    def load_vector_store_index():
        '''
        To load vector store index
        '''
        pass

    
# class LlamaPineconeVectorStore(BaseVectorStore):
#     def __init__(self, api_key: str, environment: str):
#         pinecone.init(api_key=api_key, environment=environment)

#     def create_index(self, 
#                      index_name: str,
#                      dimension: int=1536,
#                      metric: str='euclidean',
#                      pod_type: str='p1.x1'):
#         '''
#         To create pinecone index with index name
#         '''

#         pinecone.create_index(
#             index_name, dimension=dimension, metric=metric, pod_type=pod_type
#         )


#     def insert_vectors_from_nodes(self, 
#                                  llm: object, 
#                                  embed_model: object,
#                                  nodes: object,
#                                  index_name: str):
#         '''
#         To insert the documents to the pinecone vector store
#         '''

#         nodes = self.create_vector_embedding(nodes, llm, embed_model)

#         # add vectors to pinecone
#         pinecone_index = pinecone.Index(index_name)
#         vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
#         vector_store.add(nodes)

#     def load_vector_store_index(self,
#                                 index_name: str) -> VectorStoreIndex:
#         '''
#         To load pinecone vector store index
#         '''

#         vector_store = PineconeVectorStore(
#             pinecone.Index(index_name)
#         )

#         return VectorStoreIndex.from_vector_store(
#             vector_store=vector_store
#         )


class LlamaAzureCosmosDBMongoVectorStore(BaseVectorStore):
    def __init__(self, conn_string: str, db_name: str, collection_name: str):
        self.mongodb_client = pymongo.MongoClient(conn_string)
        self.store = AzureCosmosDBMongoDBVectorSearch(
            mongodb_client=self.mongodb_client,
            db_name=db_name,
            collection_name=collection_name,
        )
        logger.info(f"connected to vector Azure Cosmos DB Mongo vector store")
        print(f"connected to vector Azure Cosmos DB Mongo vector store")
    
    def insert_vectors_from_nodes(self, 
                                 llm: object, 
                                 embed_model: object,
                                 nodes: object):
        '''
        To insert the documents to the Azure Cosmos DB Mongo vector store
        '''

        nodes = self.create_vector_embedding(nodes, llm, embed_model)

        # add vectors to Azure Cosmos DB Mongo
        self.store.add(nodes)

    def load_vector_store_index(self, llm: object, embed_model: object) -> VectorStoreIndex:
        '''
        To load Azure Cosmos DB Mongo vector store index
        '''

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

        return VectorStoreIndex.from_vector_store(
            vector_store=self.store, service_context=service_context
        )

    def conn_disconnect(self):
        '''
        To disconnect the vector store
        '''
        logger.info("disconnect the vector store")
        print("disconnect the vector store")
        self.mongodb_client.close()


class LlamaAzurePostgresDBVectorStore(BaseVectorStore):
    def __init__(self, table_name: str, db_name: str, host: str, password: str, username: str):
        self.store = PGVectorStore.from_params(
            database=db_name,
            host=host,
            password=password,
            port=5432,
            user=username,
            # connection_string=conn_string,
            # async_connection_string=async_conn_string,
            # schema_name=schema_name,
            table_name=table_name,
            embed_dim=1536
        )
        logger.info(f"connected to vector Azure Postgres DB vector store")
        print(f"connected to vector Azure Postgres DB vector store")
    
    def insert_vectors_from_nodes(self, 
                                 llm: object, 
                                 embed_model: object,
                                 nodes: object):
        '''
        To insert the documents to the Azure Postgres DB vector store
        '''

        nodes = self.create_vector_embedding(nodes, llm, embed_model)

        # add vectors to Azure Postgres DB
        self.store.add(nodes)

    def load_vector_store_index(self, llm: object, embed_model: object) -> VectorStoreIndex:
        '''
        To load Azure Postgres DB vector store index
        '''

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

        return VectorStoreIndex.from_vector_store(
            vector_store=self.store, service_context=service_context
        )


# class LlamaElasticSearchVectorStore(BaseVectorStore):
#     def __init__(self, es_endpoint: str, es_cloud_id: str, es_username: str, es_password: str, index_name: str):
#         self.store = ElasticsearchStore(
#             es_url=es_endpoint,
#             index_name=index_name,
#             # es_cloud_id=es_cloud_id,
#             es_user=es_username,
#             es_password=es_password
#         )
#         logger.info(f"connected to elasticSearch DB vector store")
#         print(f"connected to elasticSearch DB vector store")
    
#     def insert_vectors_from_nodes(self, 
#                                  llm: object, 
#                                  embed_model: object,
#                                  nodes: object):
#         '''
#         To insert the documents to the elasticSearch DB vector store
#         '''

#         nodes = self.create_vector_embedding(nodes, llm, embed_model)

#         # add vectors to elasticSearch DB
#         self.store.add(nodes)

#     def load_vector_store_index(self, llm: object, embed_model: object) -> VectorStoreIndex:
#         '''
#         To load elasticSearch DB vector store index
#         '''

#         service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

#         return VectorStoreIndex.from_vector_store(
#             vector_store=self.store, service_context=service_context
#         )

