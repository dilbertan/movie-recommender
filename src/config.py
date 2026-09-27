"""Yapılandırma yükleme ve doğrulama modülü.

Bu modül `config.yaml` dosyasını okur, eksik veya hatalı parametreleri denetler
ve kullanıcıya anlaşılır Türkçe hata mesajları sunar.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigError(Exception):
    """Yapılandırma hataları için özel istisna sınıfı."""
    pass


@dataclass
class DatasetConfig:
    """Veri seti parametreleri."""
    name: str
    max_films: int
    min_year: int
    max_words: int
    random_seed: int


@dataclass
class FiltersConfig:
    """Filtreleme varsayılan parametreleri."""
    default_min_year: int
    default_max_year: int


@dataclass
class HybridConfig:
    """Hibrit arama parametreleri."""
    enabled: bool
    tfidf_weight: float


@dataclass
class PathsConfig:
    """Dosya ve klasör yolları."""
    data_dir: str
    cache_dir: str
    clean_data_file: str
    embeddings_file: str
    metadata_file: str


@dataclass
class AppConfig:
    """Tüm uygulama yapılandırmasını tutan ana veri sınıfı."""
    top_k: int
    min_similarity: float
    score_display: str
    model_name: str
    device: str
    batch_size: int
    max_seq_length: int
    dataset: DatasetConfig
    filters: FiltersConfig
    hybrid: HybridConfig
    paths: PathsConfig
    raw_dict: Dict[str, Any] = field(default_factory=dict)


def validate_config(raw: Dict[str, Any]) -> None:
    """Yapılandırma sözlüğünün geçerliliğini denetler.

    Args:
        raw: YAML dosyasından okunan ham sözlük.

    Raises:
        ConfigError: Herhangi bir zorunlu alan eksik veya geçersizse.
    """
    if not isinstance(raw, dict):
        raise ConfigError("Yapılandırma dosyası geçerli bir YAML sözlüğü içermelidir.")

    # Temel sayısal ve metin alanları
    if "top_k" not in raw or not isinstance(raw["top_k"], int) or raw["top_k"] < 1:
        raise ConfigError("'top_k' parametresi 1 veya daha büyük bir tam sayı olmalıdır.")

    if "min_similarity" not in raw or not isinstance(raw["min_similarity"], (int, float)) or not (0.0 <= raw["min_similarity"] <= 1.0):
        raise ConfigError("'min_similarity' parametresi 0.0 ile 1.0 arasında bir ondalıklı sayı olmalıdır.")

    if raw.get("score_display") not in ("raw_cosine", "minmax"):
        raise ConfigError("'score_display' değeri 'raw_cosine' veya 'minmax' olmalıdır.")

    if not raw.get("model_name") or not isinstance(raw["model_name"], str):
        raise ConfigError("'model_name' geçerli bir model adı dizgisi olmalıdır.")

    if raw.get("device") not in ("auto", "cuda", "cpu"):
        raise ConfigError("'device' değeri 'auto', 'cuda' veya 'cpu' olmalıdır.")

    if "batch_size" not in raw or not isinstance(raw["batch_size"], int) or raw["batch_size"] < 1:
        raise ConfigError("'batch_size' 1 veya daha büyük bir tam sayı olmalıdır.")

    if "max_seq_length" not in raw or not isinstance(raw["max_seq_length"], int) or raw["max_seq_length"] < 16:
        raise ConfigError("'max_seq_length' en az 16 olmalıdır.")

    # Veri seti parametreleri
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise ConfigError("'dataset' ayar bloğu eksik veya hatalı.")
    if not dataset.get("name") or not isinstance(dataset["name"], str):
        raise ConfigError("'dataset.name' boş bırakılamaz.")
    if "max_films" not in dataset or not isinstance(dataset["max_films"], int) or dataset["max_films"] < 1:
        raise ConfigError("'dataset.max_films' 1 veya daha büyük bir tam sayı olmalıdır.")
    if "min_year" not in dataset or not isinstance(dataset["min_year"], int):
        raise ConfigError("'dataset.min_year' bir yıl tam sayısı olmalıdır (ör. 1900).")
    if "max_words" not in dataset or not isinstance(dataset["max_words"], int) or dataset["max_words"] < 10:
        raise ConfigError("'dataset.max_words' en az 10 olmalıdır.")

    # Hibrit ayarları
    hybrid = raw.get("hybrid")
    if not isinstance(hybrid, dict):
        raise ConfigError("'hybrid' ayar bloğu eksik veya hatalı.")
    if not isinstance(hybrid.get("enabled"), bool):
        raise ConfigError("'hybrid.enabled' true veya false olmalıdır.")
    tfidf_w = hybrid.get("tfidf_weight", 0.2)
    if not isinstance(tfidf_w, (int, float)) or not (0.0 <= tfidf_w <= 1.0):
        raise ConfigError("'hybrid.tfidf_weight' 0.0 ile 1.0 arasında olmalıdır.")

    # Yollar
    paths = raw.get("paths")
    if not isinstance(paths, dict):
        raise ConfigError("'paths' ayar bloğu eksik veya hatalı.")
    for p_key in ("data_dir", "cache_dir", "clean_data_file"):
        if not paths.get(p_key):
            raise ConfigError(f"'paths.{p_key}' alanı boş bırakılamaz.")


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """YAML yapılandırma dosyasını yükler, doğrular ve AppConfig nesnesi döner.

    Args:
        config_path: config.yaml dosyasının yolu.

    Returns:
        AppConfig: Doğrulanmış ayar nesnesi.

    Raises:
        ConfigError: Dosya bulunamazsa, YAML biçimi bozuksa veya değerler geçersizse.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Yapılandırma dosyası bulunamadı: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Yapılandırma dosyası (YAML) okunamadı: {exc}") from exc

    validate_config(raw)

    return AppConfig(
        top_k=raw["top_k"],
        min_similarity=float(raw["min_similarity"]),
        score_display=raw["score_display"],
        model_name=raw["model_name"],
        device=raw["device"],
        batch_size=raw["batch_size"],
        max_seq_length=raw["max_seq_length"],
        dataset=DatasetConfig(
            name=raw["dataset"]["name"],
            max_films=raw["dataset"]["max_films"],
            min_year=raw["dataset"]["min_year"],
            max_words=raw["dataset"]["max_words"],
            random_seed=raw["dataset"].get("random_seed", 42),
        ),
        filters=FiltersConfig(
            default_min_year=raw.get("filters", {}).get("default_min_year", 1950),
            default_max_year=raw.get("filters", {}).get("default_max_year", 2026),
        ),
        hybrid=HybridConfig(
            enabled=raw["hybrid"]["enabled"],
            tfidf_weight=float(raw["hybrid"]["tfidf_weight"]),
        ),
        paths=PathsConfig(
            data_dir=raw["paths"]["data_dir"],
            cache_dir=raw["paths"]["cache_dir"],
            clean_data_file=raw["paths"]["clean_data_file"],
            embeddings_file=raw["paths"].get("embeddings_file", "cache/embeddings.npy"),
            metadata_file=raw["paths"].get("metadata_file", "cache/metadata.json"),
        ),
        raw_dict=raw,
    )
