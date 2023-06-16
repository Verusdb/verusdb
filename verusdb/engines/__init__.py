from __future__ import annotations
from abc import ABC, abstractmethod

class BaseEngine(ABC):


    @abstractmethod
    def add(self,collection: str, texts: list[str], embeddings: list[list[float]], metadata: list[dict[str, str]]):
        pass

    @abstractmethod
    def _serialize(self, documents):
        pass
    
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def _cosine_similarity(self, embedding: list[float], filters: dict[str, str] | None = None):
        pass
    
    @abstractmethod
    def search(self, query, filters, num_results=10) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def save(self, path) -> bool:
        pass

    