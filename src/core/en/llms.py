from abc import ABC, abstractmethod
from llama_index.llms import (
    OpenAI,
    AzureOpenAI
)
import logging


# logger
logger = logging.getLogger(__name__)


# classes for llm models
class BaseLLMModel(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model()
    
    @abstractmethod
    def _load_model(self):
        '''
        To load the llm model
        '''
        pass


class LlamaOpenAI(BaseLLMModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = self._load_model()

    def _load_model(self):
        '''
        To load the OpenAI llm model using llama-index

        Return:
            OpenAI llm model (OpenAI): It is the llm model from OpenAI
        '''

        return OpenAI(self.model_name)


class LlamaAzureOpenAI(BaseLLMModel):
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
        To load Azure OpenAI llm model using llama-index

        Return:
            Azure OpenAI model (AzureOpenAI): It is the Azure llm model from Azure OpenAI
        '''

        print("run the load_model")

        return AzureOpenAI(
            model=self.model_name,
            deployment_name=self.deployment_name,
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )