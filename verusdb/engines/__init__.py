from __future__ import annotations
from abc import ABC, abstractmethod

class BaseEngine(ABC):


    @abstractmethod
    def add(self, texts: list[str],  collection: str | None = None, embeddings: list[list[float]] | None = None, metadata: list[dict[str, str]] | None = None):
        pass

    @abstractmethod
    def _serialize(self, documents):
        pass
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def search(self, collection, query, filters, num_results=10) -> list[dict[str, str]]:
        pass

    