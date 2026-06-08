import pytest
from eval.evaluate import recall_at_k, mean_reciprocal_rank, _latency_stats


# --- recall_at_k ---


def test_recall_at_k_hit():
    assert recall_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["b", "c", "d"], {"a"}, k=3) == 0.0


def test_recall_at_k_outside_k():
    assert recall_at_k(["b", "c", "a"], {"a"}, k=2) == 0.0


def test_recall_at_k_larger_than_list():
    # Should not raise even when k > len(retrieved_ids)
    assert recall_at_k(["a"], {"a"}, k=100) == 1.0


# --- mean_reciprocal_rank ---


def test_mrr_rank_1():
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_rank_3():
    assert abs(mean_reciprocal_rank(["b", "c", "a"], {"a"}) - 1 / 3) < 1e-9


def test_mrr_not_present():
    assert mean_reciprocal_rank(["b", "c", "d"], {"a"}) == 0.0


# --- _latency_stats ---


def test_latency_stats_percentiles():
    latencies = [float(i) for i in range(1, 101)]
    stats = _latency_stats(latencies)
    assert 49.0 <= stats["p50"] <= 51.0
    assert 94.0 <= stats["p95"] <= 96.0
