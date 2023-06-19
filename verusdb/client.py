from __future__ import annotations
import polars as pl
from verusdb.settings import Settings
import numpy as np
from verusdb.engines.polars import PolarsEngine
from verusdb.engines.redis import RedisEngine


class VerusClient:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection = 'verusdb'
        
        if self.settings.engine == 'redis':
            self.engine = RedisEngine(settings)
            
        if self.settings.engine == 'polars':
            self.engine = PolarsEngine(settings)
            
            
        self.engine.load()
        
        
    def get_store(self):
        return self.engine.store
    
    def get_engine(self):
        return self.engine
        

    def add(self, texts: list[str],  collection: str | None = None, embeddings: list[list[float]] | None = None, metadata: list[dict[str, str]] | None = None):
        """
        Add a document to the dataframe
        """
        if collection is None:
            collection = self.collection
            
        self.engine.add(texts, collection, embeddings, metadata)

    def search(self, text: str| None = None, collection:str | None =  None, embedding: list[float] | None = None, filters: dict[str, str] | None = None, top_k: int = 10):
        """
        Search for similar documents
        """
        
        if text is None and embedding is None:
            raise ValueError('Either text or embedding must be provided')
        
        if text and embedding:
            raise ValueError('Only one of text or embedding must be provided')

        if text:
            return self.engine.search_text(text, collection, filters, top_k)
        
        if embedding is None:
            raise ValueError('Embedding must be provided for the search')        
    
        return self.engine.search(embedding, collection, filters, top_k)
    
    def clear(self):
        self.engine.clear()
    

    def save(self):
        """
        Save the dataframe to disk
        """
        self.engine.save()

    def delete(self, filters: dict[str, str]):
        """
        Delete documents from the dataframe
        """
        self.engine.delete(filters)

