from __future__ import annotations
from abc import ABC, abstractmethod

class BaseEmbeddingsEngine(ABC):
    """
    An abstract class that defines the interface for an embeddings engine.
    """
    
    
    @abstractmethod
    def encode(self, text: str) -> list[float]:
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        pass