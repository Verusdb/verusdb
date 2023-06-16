from __future__ import annotations
import polars as pl
from verusdb.settings import Settings
import numpy as np
from verusdb.engines.polars import PolarsEngine


class VerusClient:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = PolarsEngine(settings)
        self.engine.load()

    def get_store(self):
        return self.engine.store
        

    def add(self, collection: str, texts: list[str], embeddings: list[list[float]] | None = None, metadata: list[dict[str, str]] | None = None):
        """
        Add a document to the dataframe
        """
        self.engine.add(collection, texts, embeddings, metadata)

    def search(self, text: str| None = None, embedding: list[float] | None = None, filters: dict[str, str] | None = None, top_k: int = 10):
        """
        Search for similar documents
        """
        if not text and not embedding:
            raise ValueError('Either text or embedding must be provided')

        if text:
            return self.engine.search_text(text, filters, top_k)
            
        return self.engine.search(embedding, filters, top_k)
    

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

