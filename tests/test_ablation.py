from eval.ablation import build_config
from config import PipelineConfig


# --- flags set correctly ---


def test_build_config_sets_flags():
    base = PipelineConfig()
    variant = {"use_hybrid": True, "use_reranker": False, "use_query_expansion": True}
    cfg = build_config(base, variant)
    assert cfg.retrieval.use_hybrid is True
    assert cfg.retrieval.use_reranker is False
    assert cfg.retrieval.use_query_expansion is True


# --- base config not mutated ---


def test_build_config_does_not_mutate_base():
    base = PipelineConfig()
    original_hybrid = base.retrieval.use_hybrid
    original_reranker = base.retrieval.use_reranker
    original_expansion = base.retrieval.use_query_expansion

    variant = {
        "use_hybrid": not original_hybrid,
        "use_reranker": not original_reranker,
        "use_query_expansion": not original_expansion,
    }
    build_config(base, variant)

    assert base.retrieval.use_hybrid == original_hybrid
    assert base.retrieval.use_reranker == original_reranker
    assert base.retrieval.use_query_expansion == original_expansion


# --- independent configs ---


def test_build_config_returns_independent_configs():
    base = PipelineConfig()
    cfg_a = build_config(
        base, {"use_hybrid": True, "use_reranker": True, "use_query_expansion": True}
    )
    cfg_b = build_config(
        base, {"use_hybrid": False, "use_reranker": False, "use_query_expansion": False}
    )

    assert cfg_a.retrieval.use_hybrid != cfg_b.retrieval.use_hybrid
    assert cfg_a.retrieval.use_reranker != cfg_b.retrieval.use_reranker
    assert cfg_a.retrieval.use_query_expansion != cfg_b.retrieval.use_query_expansion

    # Mutating cfg_a's retrieval object must not affect cfg_b
    cfg_a.retrieval.use_hybrid = False
    assert cfg_b.retrieval.use_hybrid is False  # cfg_b's own value, unchanged
    cfg_a.retrieval.use_hybrid = True  # restore; assert cfg_b still independent
    assert cfg_b.retrieval.use_hybrid is False
