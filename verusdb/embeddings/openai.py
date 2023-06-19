from __future__ import annotations
import openai
import numpy as np
from verusdb.embeddings import BaseEmbeddingsEngine

class OpenAIEmbeddingsEngine(BaseEmbeddingsEngine):
    
    '''
    text-embedding-ada-002
    '''
    def __init__(self, api_key: str, api_type: str | None = None, api_base: str | None = None, api_version: str | None = None, fake: bool = False):

        self.__dimensions = 1536
        self.__fake = fake
        openai.api_key = api_key
        
        if type == 'azure':
            # TODO: add validation in case not all parameters are provided
            openai.api_type = api_type 
            openai.api_base = api_base 
            openai.api_version = api_version 

    def encode(self, text: str) -> list[float]:
        
        if self.__fake:
            return np.random.rand(self.__dimensions).tolist()
        response  = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        ) 
        return response['data'][0]['embedding'] # type: ignore

    def get_dimensions(self) -> int:
        return self.__dimensions  


