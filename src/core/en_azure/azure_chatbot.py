import openai, os
import json
from dotenv import load_dotenv
from typing import Generator, List
import re

load_dotenv()

# Azure OpenAI on your own data is only supported by the 2023-08-01-preview API version
api_type = os.getenv("OPENAI_API_TYPE")
api_version = os.getenv("OPENAI_API_VERSION")

# Azure OpenAI setup
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") # Add your endpoint here
api_key = os.getenv("AZURE_OPENAI_API_KEY") # Add your OpenAI API key here
deployment_id = os.getenv("AZURE_OPENAI_MODEL_NAME") # Add your deployment ID here
embedding_name = os.getenv("AZURE_OPENAI_EMBEDDING_NAME")

# Azure AI Search setup
search_endpoint = os.getenv("SEARCH_ENDPOINT"); # Add your Azure AI Search endpoint here
search_key = os.getenv("SEARCH_KEY"); # Add your Azure AI Search admin key here
search_index_name = os.getenv("SEARCH_NAME"); # Add your Azure AI Search index name here
search_semantic_name = os.getenv("SEARCH_SEMANTIC_NAME")

# system message
system_message = os.getenv("AZURE_OPENAI_SYSTEM_MESSAGE")

class AzureChatbot(object):
    # constructor for client
    def __init__(self, conversation: List[dict]):
        print("Azure Chatbot Initialized")
        self.client :openai.lib.azure.AzureOpenAI = openai.AzureOpenAI(
            base_url=f"{endpoint}/openai/deployments/{deployment_id}/extensions",
            api_key=api_key,
            api_version=api_version,
        )
        self.conversation :list[dict] = conversation
        self.llm_model: str = None
        # to be developed [TBD]
        self.citations :list[dict] = None

    @staticmethod
    def get_name():
        return "AzureChatbot"
    
    @staticmethod
    def preprocess_response(text):
        # This regex pattern looks for [docN] where N is any integer.
        pattern = r'\[doc(\d+)\]'
        
        # The substitution function uses the number captured in the regex.
        # It applies the subscript HTML tag to the number N.
        def subscript_replacement(match):
            return f'<sub>[{match.group(1)}]</sub>'
        
        # Perform the actual replacement using the regex pattern and substitution function.
        processed_text = re.sub(pattern, subscript_replacement, text)
        
        # Find all matches of the pattern in the string
        matches = re.findall(pattern, text)

        # Convert all matches to integers
        numbers = [int(match) for match in matches]

        # Output the maximum number if any are found, otherwise None
        max_number = max(numbers) if numbers else None
        
        return processed_text, max_number
    

    # conversation is list of dicts
    def response_stream(self) -> Generator[str, None, None]:
        for response in self.client.chat.completions.create(
                model=deployment_id,
                messages=self.conversation,
                stream=True,
                extra_body={
                    "dataSources": [
                        {
                            "type": "AzureCognitiveSearch",
                            "parameters": {
                                "endpoint": search_endpoint,
                                "key": search_key,
                                "indexName": search_index_name,
                                "inScope": True,
                                "topNDocuments": 5,
                                "queryType": "semantic",
                                "semanticConfiguration": "default",
                                "embeddingDeploymentName": embedding_name,
                                "filter": None,
                                "roleInformation": system_message,
                                "strictness": 3,
                                "semanticConfiguration": search_semantic_name,
                                "queryType": "vectorSemanticHybrid",
                                    "fieldsMapping": {
                                        "contentFieldsSeparator": "\n",
                                        "contentFields": [
                                            "content"
                                        ],
                                        "filepathField": "filename",
                                        "titleField": "title",
                                        "urlField": "url",
                                        "vectorFields": [
                                            "vector"
                                        ]
                                    }
                            }
                        }
                    ]
                }
            ):

            response_dumps = response.model_dump()
            self.llm_model = response_dumps['model']
            delta = response_dumps['choices'][0]['delta']
            if "context" in delta.keys():
                self.citations: list[dict] = json.loads(delta['context']['messages'][0]['content'])['citations']

            yield delta['content']

        # example self.citations aligned List[dict] , N citations gives N elements
        # [{'content': '8 Special Notes on Temporary Insurance Agreement (TIA)  \n \n\nTIA \n\nBackground & \nCoverage \n\n- TIA is to provide temporary life insurance coverage to the proposed insured. \n- It covers the period during which the insurance application is being underwritten \n\nby Manulife. The period starts at the time of completion of the application and \nsubject to the conditions stated in the TIA. \n\n- The amount of insurance under this agreement and all other temporary insurance \nagreements with the Company, please refer to “Terms” under TIA in details. \n\n \n\nPay the death \nbenefit \n\n- Manulife will pay the death benefit: \n➢ If the proposed insured die due to causes not related to any material \n\nmisrepresentation in the application form. \n➢ Death benefit is not payable if the cause of death is “suicide”',
        # 'filepath': 'page_040.pdf',
        # 'url': 'https://XXXXX.blob.core.windows.net/split-pdfs/page_040.pdf',
        #  },
        # {'content': '4 for details \n \n\n8 Direct Debit Authorization Form (DDA) - Selected autopay as the payment method / monthly \n\nas the payment mode \n\n- Accept agent memo to confirm DDA form will be \n\nsubmitted later \n \n\n9 Temporary Insurance Agreement (TIA) Refer to Section 3.8 for details \n \n\n10 Financial Questionnaire Refer to Section 5 to 7 for details \n \n\n11 Important Facts Statement – Policy \n\nReplacement (IFS-PR) \n\nIFS-PR is required if customer has answered “Yes” or \n“Not Yet Decided” for the “Policy Replacement” section \nof the application form. Refer to Section 3.9 for details \n \n\n12 FATCA requirements if US indicia is \n\npresent \n\nRefer to Section 11.2 for details',
        # 'filepath': 'page_009.pdf',
        # 'url': 'https://XXXXX.blob.core.windows.net/split-pdfs/page_009.pdf',
        # }]