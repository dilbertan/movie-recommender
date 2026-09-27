"""Metin embedding üretimi, model yönetimi ve disk önbellekleme modülü.

Bu modül Sentence-Transformers kullanarak film özetlerini vektörleştirir,
L2 normalize eder, GPU/CPU kaynaklarını yönetir ve önbelleğe kaydeder.
"""

import hashlib
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.config import AppConfig


def get_device(requested_device: str = "auto") -> str:
    """Çalışma donanımını belirler (GPU/CPU) ve gerektiğinde CPU'ya düşer.

    Args:
        requested_device: 'auto', 'cuda' veya 'cpu'.

    Returns:
        str: 'cuda' veya 'cpu'.
    """
    if requested_device == "cpu":
        return "cpu"

    cuda_available = torch.cuda.is_available()
    if requested_device == "cuda":
        if cuda_available:
            return "cuda"
        print("Uyarı: CUDA talep edildi ancak GPU bulunamadı. CPU kullanılacak.")
        return "cpu"

    # 'auto' modu
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU tespit edildi ve kullanılacak: {device_name} (CUDA)")
        return "cuda"

    print("GPU tespit edilemedi. İşlemler CPU üzerinde yürütülecek.")
    return "cpu"


def load_model(config: AppConfig, device: Optional[str] = None) -> SentenceTransformer:
    """Embedding modelini belirtilen veya otomatik seçilen donanımda yükler.

    Args:
        config: Uygulama yapılandırması.
        device: 'cuda', 'cpu' veya None ('auto' için).

    Returns:
        SentenceTransformer: Yüklenmiş model.
    """
    if device is None:
        device = get_device(config.device)

    print(f"Model yükleniyor: {config.model_name} (Cihaz: {device})...")
    model = SentenceTransformer(config.model_name, device=device)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = config.max_seq_length

    # GPU üzerinde fp16 (yarım duyarlık) desteği bellek tasarrufu sağlar
    if device == "cuda":
        try:
            model.half()
            print("GPU için fp16 (yarım duyarlık) modu etkinleştirildi.")
        except Exception as e:
            print(f"fp16 modu etkinleştirilemedi, standart duyarlık kullanılıyor: {e}")

    return model


def calculate_data_hash(df: pd.DataFrame) -> str:
    """Veri çerçevesinin içerik özetini (MD5 hash) hesaplar."""
    key_str = f"{len(df)}_{df['title'].iloc[0] if len(df) > 0 else ''}_{df['title'].iloc[-1] if len(df) > 0 else ''}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def get_cache_metadata(config: AppConfig, df: pd.DataFrame, embedding_dim: int) -> Dict[str, Any]:
    """Önbellek doğrulama imzası üretir."""
    return {
        "model_name": config.model_name,
        "max_films": config.dataset.max_films,
        "row_count": len(df),
        "embedding_dim": embedding_dim,
        "data_hash": calculate_data_hash(df),
    }


def is_cache_valid(config: AppConfig, df: pd.DataFrame) -> bool:
    """Mevcut önbellek dosyalarının geçerli olup olmadığını denetler.

    Model adı, film sayısı veya veri seti değiştiğinde önbellek geçersiz sayılır.
    """
    emb_path = Path(config.paths.embeddings_file)
    meta_path = Path(config.paths.metadata_file)

    if not emb_path.is_file() or not meta_path.is_file():
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)

        # Kontrol edilecek parametreler
        if cached_meta.get("model_name") != config.model_name:
            print(f"Model adı değişmiş ({cached_meta.get('model_name')} -> {config.model_name}). Önbellek yenilenecek.")
            return False

        if cached_meta.get("row_count") != len(df):
            print(f"Veri satır sayısı değişmiş ({cached_meta.get('row_count')} -> {len(df)}). Önbellek yenilenecek.")
            return False

        if cached_meta.get("data_hash") != calculate_data_hash(df):
            print("Veri içeriği değişmiş. Önbellek yenilenecek.")
            return False

        return True
    except Exception as e:
        print(f"Önbellek meta dosyası okunamadı ({e}). Önbellek yenilenecek.")
        return False


def compute_and_cache_embeddings(
    df: pd.DataFrame,
    config: AppConfig,
    force_recompute: bool = False
) -> np.ndarray:
    """Film özetlerinin embedding'lerini üretir, L2 normalize eder ve önbelleğe kaydeder.

    Args:
        df: Temizlenmiş film verisi.
        config: Uygulama yapılandırması.
        force_recompute: True ise önbellek yok sayılarak yeniden hesaplanır.

    Returns:
        np.ndarray: Normalize edilmiş embedding matrisi (N, D).
    """
    emb_path = Path(config.paths.embeddings_file)
    meta_path = Path(config.paths.metadata_file)

    # Önbellek geçerliyse doğrudan diskten yükle
    if not force_recompute and is_cache_valid(config, df):
        print(f"Geçerli önbellek bulundu, yükleniyor: {emb_path}")
        embeddings = np.load(emb_path)
        # Boyut kontrolü
        if embeddings.shape[0] == len(df):
            print(f"Önbellekten {embeddings.shape[0]} film embedding'i başarıyla yüklendi (Boyut: {embeddings.shape[1]}).")
            return embeddings
        print("Önbellek satır sayısı ile veri uyuşmuyor, yeniden hesaplanıyor...")

    # Önbellek dizinini oluştur
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    overviews = df["overview"].tolist()
    device = get_device(config.device)
    model = load_model(config, device=device)

    print(f"Toplam {len(overviews)} film özeti için embedding üretiliyor (Batch size: {config.batch_size})...")

    try:
        embeddings = model.encode(
            overviews,
            batch_size=config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except Exception as exc:
        # GPU Out of Memory (OOM) veya benzeri durumda CPU'ya çekilme
        if device == "cuda":
            print(f"GPU işlemi sırasında hata alındı ({exc}). Otomatik olarak CPU'ya geçiliyor...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpu_model = load_model(config, device="cpu")
            embeddings = cpu_model.encode(
                overviews,
                batch_size=max(1, config.batch_size // 2),
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            raise exc

    embeddings = embeddings.astype(np.float32)

    # Diske kaydet
    np.save(emb_path, embeddings)
    metadata = get_cache_metadata(config, df, embeddings.shape[1])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Embedding'ler başarıyla kaydedildi: {emb_path} (Şekil: {embeddings.shape})")
    return embeddings
