"""Veri temizleme ve işleme modülü testleri."""

import pandas as pd
import pytest
from src.config import load_config
from src.data import (
    parse_year,
    clean_text,
    truncate_words,
    clean_genres,
    process_movies_dataframe,
)


def test_parse_year():
    """Çeşitli tarih formatlarından yıl çıkarma testi."""
    assert parse_year("1994-09-23") == 1994
    assert parse_year("2008") == 2008
    assert parse_year(1972) == 1972
    assert parse_year("15 December 1993") == 1993
    assert parse_year(None) is None
    assert parse_year("") is None
    assert parse_year("Geçersiz Tarih") is None


def test_truncate_words():
    """Kelime sayısı sınırlama testi."""
    text = "word " * 10
    truncated = truncate_words(text, 5)
    assert len(truncated.split()) == 5
    assert truncated.endswith("...")
    assert truncate_words(text, 20) == text.strip()


def test_clean_genres():
    """Tür metinlerini standartlaştırma testi."""
    assert clean_genres("Action, Crime") == "Action, Crime"
    assert clean_genres("['Drama', 'Thriller']") == "Drama, Thriller"
    assert clean_genres("") == "Bilinmiyor"
    assert clean_genres(None) == "Bilinmiyor"


def test_process_movies_dataframe():
    """Tüm temizleme ve filtreleme akışını sahte veri ile test eder."""
    config = load_config("config.yaml")

    fake_data = pd.DataFrame([
        {
            "title": "Movie Long Overview",
            "overview": "This is a detailed overview of a movie that has more than twenty words in total so it passes the minimum word count filter easily.",
            "release_date": "2000-01-01",
            "genre": "Drama",
            "popularity": 50.0,
            "vote_average": 8.0,
        },
        {
            "title": "Movie Short Overview",
            "overview": "Too short.",  # < 20 kelime -> Atılmalı!
            "release_date": "2005-01-01",
            "genre": "Comedy",
            "popularity": 40.0,
            "vote_average": 7.0,
        },
        {
            "title": "Movie Old",
            "overview": "This is a very old movie from 1850 that has more than twenty words in its overview so it passes the word length filter.",
            "release_date": "1850-01-01",  # < min_year (1900) -> Atılmalı!
            "genre": "History",
            "popularity": 10.0,
            "vote_average": 5.0,
        },
        {
            "title": "Duplicate Movie",
            "overview": "This is the first copy of a duplicate movie with a sufficiently long overview for our movie recommendation test suite.",
            "release_date": "2010-01-01",
            "genre": "Sci-Fi",
            "popularity": 80.0,  # Daha yüksek popülerlik -> Bu tutulmalı
            "vote_average": 8.5,
        },
        {
            "title": "Duplicate Movie",
            "overview": "This is the second copy of a duplicate movie with a sufficiently long overview for our movie recommendation test suite.",
            "release_date": "2010-01-01",
            "genre": "Sci-Fi",
            "popularity": 20.0,  # Düşük popülerlik -> Atılmalı
            "vote_average": 6.0,
        },
    ])

    clean_df = process_movies_dataframe(fake_data, config)

    # 1) Kısa özet atıldı mı?
    assert "Movie Short Overview" not in clean_df["title"].values

    # 2) 1850 yapımı eski film atıldı mı?
    assert "Movie Old" not in clean_df["title"].values

    # 3) Yinelenen film tekil kaldı mı ve yüksek popülerlikli olan mı seçildi?
    dup_rows = clean_df[clean_df["title"] == "Duplicate Movie"]
    assert len(dup_rows) == 1
    assert dup_rows.iloc[0]["popularity"] == 80.0

    # 4) Geçerli film listede mi?
    assert "Movie Long Overview" in clean_df["title"].values

    # 5) display_title düzgün oluşturuldu mu?
    assert "Movie Long Overview (2000)" in clean_df["display_title"].values
