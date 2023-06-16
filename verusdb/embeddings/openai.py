from __future__ import annotations
import openai
from verusdb.embeddings import EmbeddingsEngine

class OpenAIEmbeddingsEngine(EmbeddingsEngine):
    
    '''
    text-embedding-ada-002
    '''
    def __init__(self, key: str, type: str | None = None, base: str | None = None, version: str | None = None):

        openai.api_key = key
        if type == 'azure':
            # TODO: add validation in case not all parameters are provided
            openai.api_type = api_type 
            openai.api_base = api_base 
            openai.api_version = api_version 

    def encode(self, text: str) -> list[float]:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']

        


