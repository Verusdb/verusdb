# Description: Utility functions for the VerusDB project
from __future__ import annotations
import uuid

def generate_uuid(dimension: int | None = None) -> str | list[str]:
    if dimension is not None:
        return [str(uuid.uuid4()) for _ in range(dimension)]
    
    return str(uuid.uuid4())