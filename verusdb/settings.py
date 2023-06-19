from __future__ import annotations
import os
import polars as pl
from abc import ABC, abstractmethod
from verusdb.embeddings import BaseEmbeddingsEngine

class Settings:
    """
    Settings class for verusdb
    """
    def __init__(self, folder: str | None = None, engine: str | None = None,  embeddings: BaseEmbeddingsEngine | None = None, **kwargs):

        self.folder = folder
        self.engine = engine
        
        
        self.embeddings = embeddings
        
        
        self.persist = False

        # validate the engine
        if self.engine not in ['polars', 'redis']:
            raise ValueError('Invalid engine')
        
        if self.engine == 'redis':
            redis = kwargs.get('redis', None)
            if redis is None:
                raise ValueError('Redis engine requires redis settings')
            
            self.redis_host = redis.get('host', 'localhost')
            self.redis_port = redis.get('port', 6379)
            self.redis_db = redis.get('db', 0)
            self.redis_password = redis.get('password', None)
            self.redis_doc_prefix = redis.get('prefix', 'doc:')
            self.redis_index = redis.get('index', 'verusdb')


        if self.folder is not None:
            self.file = self.folder+'/verusdb.parquet'
            self.persist = True

    def get_file(self):
        return self.file
    
    


    

    