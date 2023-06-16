from __future__ import annotations
from abc import ABC, abstractmethod

class EmbeddingsEngine(ABC):
    """
    An abstract class that defines the interface for an embeddings engine.
    """
    @abstractmethod
    def encode(self, text: str) -> list[float]:
        pass