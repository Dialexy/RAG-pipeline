from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any]
