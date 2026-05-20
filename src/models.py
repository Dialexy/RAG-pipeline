from dataclasses import dataclass
from typing import Any
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any]

    def __hash__(self) -> int:
        return hash(self.id)
