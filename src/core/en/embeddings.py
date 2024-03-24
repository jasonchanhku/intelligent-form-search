from abc import ABC, abstractmethod
from llama_index.embeddings import (
    OpenAIEmbedding, 
    AzureOpenAIEmbedding
)
import logging


# logger
logger = logging.getLogger(__name__)


# classes for llm embedding models
class BaseLLmEmbeddingModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model()
    
    @abstractmethod
    def _load_model(self):
        '''
        To load the llm embedding model
        '''
        pass


class LlamaOpenAIEmbedding(BaseLLmEmbeddingModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.load_model()
        self.model = self._load_model()

    def _load_model(self):
        '''
        To load the OpenAI embedding model using llama-index

        Return:
            OpenAI embedding model (OpenAIEmbedding): It is the embedding model from OpenAI
        '''

        return OpenAIEmbedding()


class LlamaAzureOpenAIEmbedding(BaseLLmEmbeddingModel):
    def __init__(self,
                 model_name: str,
                 deployment_name: str,
                 api_key: str,
                 azure_endpoint: str,
                 api_version: str):
        self.model_name = model_name
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.model = self._load_model()

    def _load_model(self):
        '''
        To load the Azure OpenAI embedding model using llama-index

        Return:
            Azure OpenAI embedding model (AzureOpenAIEmbedding): It is the embedding model from Azure OpenAI
        '''

        return AzureOpenAIEmbedding(
            model=self.model_name,
            deployment_name=self.deployment_name,
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )