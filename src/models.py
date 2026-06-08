from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any]

    def __hash__(self) -> int:
        return hash(self.id)
