"""Yapılandırma modülü testleri."""

import pytest
from pathlib import Path
from src.config import load_config, validate_config, ConfigError


def test_load_default_config():
    """Varsayılan config.yaml dosyasının başarıyla yüklendiğini doğrular."""
    config = load_config("config.yaml")
    assert config.top_k >= 1
    assert 0.0 <= config.min_similarity <= 1.0
    assert config.score_display in ("raw_cosine", "minmax")
    assert config.model_name != ""
    assert config.device in ("auto", "cuda", "cpu")
    assert config.dataset.max_films > 0
    assert config.paths.clean_data_file.endswith(".parquet")


def test_missing_config_file():
    """Var olmayan bir dosya yolunda ConfigError fırlatıldığını doğrular."""
    with pytest.raises(ConfigError, match="bulunamadı"):
        load_config("non_existent_config_file_12345.yaml")


def test_invalid_top_k():
    """Geçersiz top_k değerinin hata verdiğini doğrular."""
    invalid_dict = {
        "top_k": 0,
        "min_similarity": 0.0,
        "score_display": "raw_cosine",
        "model_name": "test-model",
        "device": "auto",
        "batch_size": 16,
        "max_seq_length": 128,
        "dataset": {"name": "test", "max_films": 100, "min_year": 1900, "max_words": 100},
        "hybrid": {"enabled": False, "tfidf_weight": 0.2},
        "paths": {"data_dir": "data", "cache_dir": "cache", "clean_data_file": "data/test.parquet"}
    }
    with pytest.raises(ConfigError, match="top_k"):
        validate_config(invalid_dict)


def test_invalid_score_display():
    """Geçersiz score_display değerinin hata verdiğini doğrular."""
    invalid_dict = {
        "top_k": 5,
        "min_similarity": 0.0,
        "score_display": "invalid_mode",
        "model_name": "test-model",
        "device": "auto",
        "batch_size": 16,
        "max_seq_length": 128,
        "dataset": {"name": "test", "max_films": 100, "min_year": 1900, "max_words": 100},
        "hybrid": {"enabled": False, "tfidf_weight": 0.2},
        "paths": {"data_dir": "data", "cache_dir": "cache", "clean_data_file": "data/test.parquet"}
    }
    with pytest.raises(ConfigError, match="score_display"):
        validate_config(invalid_dict)
