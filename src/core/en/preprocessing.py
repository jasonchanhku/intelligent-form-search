from abc import ABC, abstractmethod
from llama_index import SimpleDirectoryReader
from llama_index.text_splitter import SentenceSplitter
import logging


# logger
logging.basicConfig(format='%(asctime)s %(clientip)-15s %(user)-8s %(message)s')
logger = logging.getLogger(__name__)


# classes for preprocessing
class BaseNodeParser(ABC):
    def __init__(self, file_path: str, extension: str):
        self.file_path = file_path
        self.extension = extension

    @abstractmethod
    def parse(self):
        '''
        To parse the documents and parse them into nodes
        '''
        pass


class LlamaSimpleNodeParser(BaseNodeParser):
    def __init__(self, file_path: str, extension: str, chunk_size: int, chunk_overlap: int):
        super().__init__(file_path, extension)
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        loader = SimpleDirectoryReader(
            input_dir=file_path,
            required_exts=[f".{extension}"],
            recursive=True
        )
        self.documents = loader.load_data()
        
    def parse(self) -> object:
        '''
        To parse the documents with SentenceSplitter and SimpleNodeParser and
        convert them into nodes for vector storage

        Returns:
            nodes (object): The nodes for storing in vector_store
        '''

        logger.info('initialize SentenceSplitter')
        sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        logger.info('create nodes from documents')
        nodes = sentence_splitter.get_nodes_from_documents(self.documents)

        return nodes