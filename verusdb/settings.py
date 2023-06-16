from __future__ import annotations
import os
import polars as pl
from abc import ABC, abstractmethod


class Settings:
    """
    Settings class for verusdb
    """
    def __init__(self, folder: str | None = None, engine: str | None = None, store: str | None = None, embeddings: str | None = None):

        self.folder = folder
        self.engine = 'polars'
        self.store = 'parquet'
        self.embeddings = embeddings
        self.persist = False

        # validate the engine
        if self.engine not in ['polars']:
            raise ValueError('Invalid engine')
            
        # validate the store
        if self.store not in ['parquet']:
            raise ValueError('Invalid store')

        if self.folder is not None:
            self.file = os.path.join(folder, 'verusdb.parquet')
            self.persist = True

    def get_file(self):
        return self.file


    

    