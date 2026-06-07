from pathlib import Path
from src.pipeline import file_hash, corpus_hash, config_fingerprint
from config import PipelineConfig, ChunkConfig


# --- file_hash ---

def test_file_hash_different_contents_differ(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello world")
    b.write_text("goodbye world")
    assert file_hash(a) != file_hash(b)


def test_file_hash_same_content_matches(tmp_path):
    # file_hash includes the filename, so use the same name in different dirs
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    a = d1 / "doc.txt"
    c = d2 / "doc.txt"
    a.write_text("hello world")
    c.write_text("hello world")
    assert file_hash(a) == file_hash(c)


# --- corpus_hash ---

def test_corpus_hash_deterministic(tmp_path):
    (tmp_path / "one.txt").write_text("alpha")
    (tmp_path / "two.txt").write_text("beta")
    assert corpus_hash(tmp_path) == corpus_hash(tmp_path)


def test_corpus_hash_changes_on_new_file(tmp_path):
    (tmp_path / "one.txt").write_text("alpha")
    (tmp_path / "two.txt").write_text("beta")
    before = corpus_hash(tmp_path)
    (tmp_path / "three.txt").write_text("gamma")
    assert corpus_hash(tmp_path) != before


def test_corpus_hash_changes_on_modified_content(tmp_path):
    f = tmp_path / "one.txt"
    f.write_text("alpha")
    (tmp_path / "two.txt").write_text("beta")
    before = corpus_hash(tmp_path)
    f.write_text("CHANGED")
    assert corpus_hash(tmp_path) != before


# --- config_fingerprint ---

def test_config_fingerprint_defaults_equal():
    assert config_fingerprint(PipelineConfig()) == config_fingerprint(PipelineConfig())


def test_config_fingerprint_differs_on_chunk_size():
    default = config_fingerprint(PipelineConfig())
    modified = config_fingerprint(PipelineConfig(chunking=ChunkConfig(chunk_size=999)))
    assert default != modified
