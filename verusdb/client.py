from __future__ import annotations
import polars as pl
from verusdb.settings import Settings
import numpy as np
from verusdb.engines.polars import PolarsEngine
from verusdb.engines.redis import RedisEngine
from verusdb.engines.postgresql import PostgreSQLEngine


class VerusClient:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection = 'verusdb'
        
        if self.settings.engine == 'redis':
            self.engine = RedisEngine(settings)
            
        if self.settings.engine == 'polars':
            self.engine = PolarsEngine(settings)
            
        if self.settings.engine == 'postgres':
            self.engine = PostgreSQLEngine(settings)
            
            
        self.engine.load()
        

    
    def get_engine(self):
        return self.engine
        

    def add(self, texts: list[str],  collection: str | None = None, embeddings: list[list[float]] | None = None, metadata: list[dict[str, str]] | None = None):
        """
        Add a document to the dataframe
        """
        if collection is None:
            collection = self.collection
            
        self.engine.add(texts=texts, collection=collection, embeddings=embeddings, metadata=metadata)

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
    
    def update(self, uuid: str, metadata: dict[str, str]):
        return self.engine.update(uuid, metadata)
        

    def get_documents(self, collection: str | None = None):
        if collection is None:
            collection = self.collection
        return self.engine.get_documents(collection=collection)
    
    def get_document(self, uuid: str):
        return self.engine.get_document(uuid)

    def save(self):
        """
        Save the dataframe to disk
        """
        if self.settings.engine == 'polars':
            self.engine.save()
        else:
            raise NotImplementedError('Save is not implemented for this engine')

    def delete(self, uuid: str| None = None, collection: str | None = None ,filters: dict[str, str]| None = None):
        """
        Delete documents from the dataframe
        """
        self.engine.delete(uuid=uuid, collection=collection, filters=filters)

