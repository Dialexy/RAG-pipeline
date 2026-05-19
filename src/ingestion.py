"""
Fetch and clean the raw corpus. Outputs plain-text documents to data/raw/.
"""

from pathlib import Path
from collections.abc import Iterator
from itertools import chain
from datetime import datetime
from typing import cast, Any
from datasets import load_dataset
from .models import Document


def fetch_corpus(output_dir: Path) -> None:
    """Download / export the chosen dataset into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.en", streaming=True, split="train"
    )

    for i, indexedfile in enumerate(dataset):
        if i >= 1000:
            break
        item = cast(dict[str, Any], indexedfile)

        filepath = output_dir / f"{i}.txt"
        if not filepath.exists():
            combined = item["title"] + "\n\n" + item["text"]
            filepath.write_text(combined, encoding="utf-8")


def iter_documents(raw_dir: Path) -> Iterator[Document]:
    """
    Yield Document objects with id, text, and metadata.
    Called by the chunking stage.
    """
    patterns = ("*.txt", "*.md")

    files = chain.from_iterable(raw_dir.rglob(p) for p in patterns)

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        metadata = {
            "fileName": file_path.name,
            "extension": file_path.suffix,
            "path": str(file_path),
            "last_modified": datetime.fromtimestamp(
                file_path.stat().st_mtime
            ).isoformat(),
        }

        yield Document(
            id=str(file_path.relative_to(raw_dir)),
            text=text,
            metadata=metadata,
        )
