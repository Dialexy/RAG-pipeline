import pytest
from src.retrieval import reciprocal_rank_fusion, bm25_search
from src.models import Document


def make_doc(text: str, doc_id: str = None) -> dict:
    return {"text": text, "metadata": {}, "score": 1.0}


# --- reciprocal_rank_fusion ---

def test_rrf_single_list_preserves_order():
    docs = [make_doc("alpha"), make_doc("beta"), make_doc("gamma")]
    result = reciprocal_rank_fusion([docs])
    texts = [r["text"] for r in result]
    assert texts == ["alpha", "beta", "gamma"]


def test_rrf_overlapping_doc_scores_higher():
    shared = make_doc("shared")
    only_a = make_doc("only_a")
    only_b = make_doc("only_b")

    list_a = [shared, only_a]
    list_b = [shared, only_b]

    result = reciprocal_rank_fusion([list_a, list_b])
    score_map = {r["text"]: r["score"] for r in result}

    assert score_map["shared"] > score_map["only_a"]
    assert score_map["shared"] > score_map["only_b"]


# --- bm25_search ---

CORPUS = [
    Document(id="1", text="the cat sat on the mat", metadata={}),
    Document(id="2", text="dogs are loyal and friendly animals", metadata={}),
    Document(id="3", text="python is a popular programming language", metadata={}),
    Document(id="4", text="the weather is sunny and warm today", metadata={}),
    Document(id="5", text="xylophone is a rare and exotic instrument", metadata={}),
]


def test_bm25_unique_term_ranks_first():
    results = bm25_search("xylophone", CORPUS, top_k=5)
    assert results[0]["text"] == "xylophone is a rare and exotic instrument"


def test_bm25_top_k_limits_results():
    results = bm25_search("the", CORPUS, top_k=2)
    assert len(results) == 2
