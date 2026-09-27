"""Film verisi indirme, temizleme ve işleme modülü.

Bu modül Hugging Face üzerinden açık veri setini çeker, temizler,
filtreler ve `data/movies_clean.parquet` dosyasına kaydeder.
"""

import os
import re
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

from src.config import AppConfig


def parse_year(val: any) -> Optional[int]:
    """Çeşitli tarih formatlarından (örn. '1994-09-23', '1994', 1994) 4 basamaklı yılı çıkarır."""
    if val is None or pd.isna(val):
        return None
    val_str = str(val).strip()
    match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", val_str)
    if match:
        return int(match.group(1))
    return None


def clean_text(text: any) -> str:
    """Metindeki gereksiz boşlukları ve kontrol karakterlerini temizler."""
    if text is None or pd.isna(text):
        return ""
    s = str(text).strip()
    # Çoklu boşlukları ve yeni satırları düzelt
    s = re.sub(r"\s+", " ", s)
    return s


def truncate_words(text: str, max_words: int) -> str:
    """Metni belirtilen kelime sayısına göre kırpar."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def clean_genres(val: any) -> str:
    """Tür alanını standart virgüllü metin formatına dönüştürür."""
    if val is None or pd.isna(val):
        return "Bilinmiyor"
    val_str = str(val).strip()
    if not val_str:
        return "Bilinmiyor"
    # '["Action", "Drama"]' gibi JSON veya liste biçimindeyse
    val_str = re.sub(r"[\[\]'\"\{\}]", "", val_str)
    parts = [p.strip() for p in re.split(r"[,/|]", val_str) if p.strip()]
    return ", ".join(parts) if parts else "Bilinmiyor"


def process_movies_dataframe(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Ham film veri çerçevesini kurallara göre temizler ve yapılandırır.

    Kurallar:
    1. Boş veya 20 kelimeden kısa özetleri atar.
    2. Yinelenen (başlık + yıl) filmleri eler (varsa daha popüler olanı korur).
    3. Özetleri config.dataset.max_words sınırına kırpar.
    4. min_year filtresini uygular.
    5. Popülerliğe göre sıralayıp veya sabit tohumla örnekleyip max_films kadar kayıt seçer.
    """
    df = df.copy()

    # Sütun isimlerini esnek bir şekilde eşle (farklı veri setleri için tolerans)
    col_map = {}
    for col in df.columns:
        c_low = col.lower().strip()
        if c_low in ("title", "movie_title", "film_title", "original_title"):
            col_map[col] = "title"
        elif c_low in ("overview", "plot", "plotsummary", "summary", "description"):
            col_map[col] = "overview"
        elif c_low in ("genre", "genres"):
            col_map[col] = "genre"
        elif c_low in ("release_date", "year", "release year"):
            col_map[col] = "release_date"
        elif c_low in ("popularity", "pop"):
            col_map[col] = "popularity"
        elif c_low in ("vote_average", "rating", "score"):
            col_map[col] = "vote_average"
        elif c_low in ("original_language", "language", "lang"):
            col_map[col] = "language"
        elif c_low in ("id", "movie_id"):
            col_map[col] = "movie_id"

    df = df.rename(columns=col_map)

    # Başlık ve özet alanlarının varlığını kontrol et
    if "title" not in df.columns or "overview" not in df.columns:
        raise ValueError("Veri setinde gerekli 'title' ve 'overview' sütunları bulunamadı!")

    # Metin temizlikleri
    df["title"] = df["title"].apply(clean_text)
    df["overview"] = df["overview"].apply(clean_text)

    # Boş başlıkları ele
    df = df[df["title"].str.len() > 0]

    # Kelime sayısı hesabı ve min ~20 kelime filtresi
    df["word_count"] = df["overview"].apply(lambda s: len(s.split()))
    df = df[df["word_count"] >= 20]

    # Yıl ayrıştırma
    if "release_date" in df.columns:
        df["year"] = df["release_date"].apply(parse_year)
    else:
        df["year"] = None

    # min_year filtresi (yılı belli olanlar için)
    min_year = config.dataset.min_year
    df = df[df["year"].isna() | (df["year"] >= min_year)]

    # Tür temizliği
    if "genre" in df.columns:
        df["genres"] = df["genre"].apply(clean_genres)
    else:
        df["genres"] = "Bilinmiyor"

    # Popülerlik ve puan
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    else:
        df["popularity"] = 0.0

    if "vote_average" in df.columns:
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    else:
        df["vote_average"] = 0.0

    if "language" not in df.columns:
        df["language"] = "en"
    else:
        df["language"] = df["language"].fillna("en").astype(str)

    # Yinelenen başlık + yıl kayıtlarını ayıkla (popülerliğe göre en yükseğini tut)
    df["clean_title_key"] = df["title"].str.lower().str.strip()
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["clean_title_key", "year"], keep="first")

    # max_words kırpması
    max_words = config.dataset.max_words
    df["overview"] = df["overview"].apply(lambda text: truncate_words(text, max_words))

    # max_films sınırlandırması
    max_films = config.dataset.max_films
    if len(df) > max_films:
        # Eğer popülerlik sıfır değilse en popülerleri seç, yoksa sabit tohumla örnekle
        if (df["popularity"] > 0).any():
            df = df.head(max_films)
        else:
            df = df.sample(n=max_films, random_state=config.dataset.random_seed)

    # Görüntüleme başlığı: "Film Adı (Yıl)" - aynı adlı filmleri ayırt etmeyi sağlar
    df["display_title"] = df.apply(
        lambda r: f"{r['title']} ({int(r['year'])})" if pd.notna(r['year']) else r['title'],
        axis=1
    )

    # Temiz indeks ve nihai sütun seçimi
    df = df.reset_index(drop=True)
    df["film_index"] = df.index

    selected_cols = [
        "film_index", "title", "year", "genres", "overview",
        "popularity", "vote_average", "language", "display_title"
    ]
    return df[selected_cols]


def download_huggingface_dataset(dataset_name: str) -> pd.DataFrame:
    """Hugging Face üzerinden veri setini çeker ve pandas DataFrame olarak döndürür."""
    from datasets import load_dataset

    print(f"Hugging Face veri seti indiriliyor: {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split="train")
        return ds.to_pandas()
    except Exception as e:
        raise RuntimeError(
            f"Hugging Face veri seti indirilemedi ({dataset_name}). "
            f"Lütfen internet bağlantınızı kontrol edin. Hata ayrıntısı: {e}"
        ) from e


def get_or_create_clean_data(config: AppConfig, force_reload: bool = False) -> pd.DataFrame:
    """Temizlenmiş film verisini yükler; yoksa indirir, temizler ve kaydeder.

    Args:
        config: Uygulama yapılandırması.
        force_reload: True ise var olan dosya yok sayılarak yeniden indirilir.

    Returns:
        pd.DataFrame: Temizlenmiş film veri tablosu.
    """
    clean_path = Path(config.paths.clean_data_file)

    if clean_path.is_file() and not force_reload:
        print(f"Önceden temizlenmiş film verisi yükleniyor: {clean_path}")
        df = pd.read_parquet(clean_path)
        return df

    # Veri dizinini oluştur
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    # Ham veriyi indir
    raw_df = download_huggingface_dataset(config.dataset.name)

    # Temizle ve işle
    print("Film verileri temizleniyor ve filtreleniyor...")
    clean_df = process_movies_dataframe(raw_df, config)

    # Parquet olarak kaydet
    clean_df.to_parquet(clean_path, index=False)
    print(f"Temizlenmiş {len(clean_df)} film başarıyla kaydedildi: {clean_path}")

    return clean_df
