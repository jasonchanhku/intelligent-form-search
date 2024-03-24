from typing import Dict, List, Any, Generator
import nest_asyncio
from src.settings import (
    db_name, collection_name,
    embed_model, embed_deployment_name, azure_resource, azure_api_version,
    llm_model, deployment_name,
    pg_table_name, pg_db_name, pg_host, pg_username, pg_password,
    conn_string, azure_openai_key,
    chatbot_config
)
from llama_index import ServiceContext, get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import (
    get_response_synthesizer,
)
from src.core.en.vector_stores import LlamaAzureCosmosDBMongoVectorStore, LlamaAzurePostgresDBVectorStore
from src.core.en.llms import LlamaAzureOpenAI
from src.core.en.embeddings import LlamaAzureOpenAIEmbedding
import logging


# logger
logger = logging.getLogger(__name__)


# async
nest_asyncio.apply()



# settings
pipe_config = {
    'vector_store': {
        'LlamaAzureCosmosDBMongo': LlamaAzureCosmosDBMongoVectorStore(
            conn_string=conn_string,
            db_name=db_name,
            collection_name=collection_name
        ),
        'LlamaAzurePostgresDB': LlamaAzurePostgresDBVectorStore(
            table_name=pg_table_name,
            db_name=pg_db_name,
            host=pg_host,
            password=pg_password,
            username=pg_username
        ),
        # 'LlamaElasticSearch': LlamaElasticSearchVectorStore(
        #     es_endpoint=es_endpoint,
        #     es_cloud_id=es_cloud_id,
        #     es_username='elastic',
        #     es_password=es_password,
        #     index_name=es_index_name
        # ),
    },
    'embed_model': {
        'LlamaAzureOpenAIEmbedding': LlamaAzureOpenAIEmbedding(
            model_name=embed_model,
            deployment_name=embed_deployment_name,
            api_key=azure_openai_key,
            azure_endpoint=f"https://{azure_resource}.openai.azure.com",
            api_version=azure_api_version
        ).model
    },
    'llm': {
        'LlamaAzureOpenAI': LlamaAzureOpenAI(
            model_name=llm_model,
            deployment_name=deployment_name,
            api_key=azure_openai_key,
            azure_endpoint=f"https://{azure_resource}.openai.azure.com",
            api_version=azure_api_version
        ).model
    }
}    


# build the query engine
rag_custom_llm = pipe_config['llm'][chatbot_config['choice'][2]]
rag_custom_embed_model = pipe_config['embed_model'][chatbot_config['choice'][1]]

service_context = ServiceContext.from_defaults(llm=rag_custom_llm, embed_model=rag_custom_embed_model)
response_synthesizer = get_response_synthesizer(streaming=True, response_mode="compact", use_async=False, service_context=service_context)

rag_custom_vector_store = pipe_config['vector_store'][chatbot_config['choice'][0]]
retriever = rag_custom_vector_store.load_vector_store_index(llm=rag_custom_llm, embed_model=rag_custom_embed_model).as_retriever()

# query_engine = RAGCustomQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
#     llm=rag_custom_llm,
#     qa_prompt=qa_prompt
# )

# query_engine = rag_custom_vector_store.load_vector_store_index(llm=rag_custom_llm, embed_model=rag_custom_embed_model).as_query_engine(streaming=True)

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
    use_async=False,
    streaming=True,
    verbose=True
)
# query_engine = rag_custom_vector_store.load_vector_store_index(llm=rag_custom_llm, embed_model=rag_custom_embed_model).as_query_engine()

response_synthesizer_eval = get_response_synthesizer(response_mode="compact", use_async=True, service_context=service_context)
query_engine_eval = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=response_synthesizer_eval,
    service_context=service_context
)

query_engine_eval_v2 = rag_custom_vector_store.load_vector_store_index(llm=rag_custom_llm, embed_model=rag_custom_embed_model).as_query_engine()

# Chatbot objects for Chatbot factory
class LlamaIndexChatbot:
    def __init__(self, conversation: List[dict]):
        self.conversation: List[dict] = conversation
        self.llm_model: str = None
        self.citations: List[dict] = None
    
    @staticmethod
    def get_name():
        return "LlamaIndexChatbot"

    def preprocess_response(self, text: str) -> str:

        suffix_text = ''
        for i in range(1, len(self.citations)+1):
            suffix_text += f'<sub>[{i}]</sub>'
        
        return text + suffix_text

    @staticmethod
    def get_citation(query_str: str) -> List[Dict[str, Any]]:
        '''
        To get a list of dictionary of citation

        Params:
            query_str (str): it is the content from self.conversation

        Return:
            citations (List[Dict[str, Any]]): It is a list of dictionary of citation
        '''

        citations = []

        retrieved_nodes = retriever.retrieve(query_str)

        for node in retrieved_nodes:
            d_citation = {}
            node = node.to_dict()
            d_citation['content'] = node['node']['text']
            d_citation['filepath'] = node['node']['metadata']['file_name']
            d_citation['url'] = node['node']['metadata']['file_path'] # need to change to storage blob link later in production

            citations.append(d_citation)
        
        return citations
       
    def response_stream(self) -> Generator[str, None, None]:
        '''
        To return the response by streaming

        Yield:
            content

        '''

        query_str = self.conversation[-1]['content']

        self.citations = self.get_citation(query_str)

        for text in query_engine.query(query_str).response_gen:
            yield text 

