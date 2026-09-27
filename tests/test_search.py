"""Arama ve öneri motoru testleri."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.search import (
    chunk_text,
    fuzzy_find_movies,
    build_filter_mask,
    search_recommendations,
    search_by_movie,
)


def test_chunk_text():
    """Uzun metin parçalama (chunking) testi."""
    text = "word " * 300
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # Kısa metin parçalanmamalı
    short_text = "This is a short plot."
    assert chunk_text(short_text, chunk_size=50) == [short_text]


def test_fuzzy_find_movies():
    """Yazım hatası toleranslı film arama testi."""
    df = pd.DataFrame([
        {"film_index": 0, "title": "The Godfather", "display_title": "The Godfather (1972)", "year": 1972, "genres": "Crime, Drama"},
        {"film_index": 1, "title": "The Godfather Part II", "display_title": "The Godfather Part II (1974)", "year": 1974, "genres": "Crime, Drama"},
        {"film_index": 2, "title": "GoodFellas", "display_title": "GoodFellas (1990)", "year": 1990, "genres": "Crime, Drama"},
    ])

    matches = fuzzy_find_movies("Godfahter", df, limit=2)
    assert len(matches) > 0
    # İlk sırada The Godfather olmalı
    assert "Godfather" in matches[0]["title"]


def test_build_filter_mask():
    """Aday havuzu filtreleme maskesi testi."""
    df = pd.DataFrame([
        {"film_index": 0, "year": 1995, "genres": "Crime, Thriller"},
        {"film_index": 1, "year": 2010, "genres": "Sci-Fi, Action"},
        {"film_index": 2, "year": 1980, "genres": "Comedy, Romance"},
    ])

    # Yıl filtresi: 1990 - 2015
    mask_year = build_filter_mask(df, min_year=1990, max_year=2015)
    assert np.array_equal(mask_year, [True, True, False])

    # Tür filtresi: 'Crime'
    mask_genre = build_filter_mask(df, genre="Crime")
    assert np.array_equal(mask_genre, [True, False, False])

    # Hariç tutma: film_index = 0
    mask_ex = build_filter_mask(df, exclude_index=0)
    assert np.array_equal(mask_ex, [False, True, True])


def test_search_and_exclude_query_movie():
    """Sorgulanan filmin sonuçlardan hariç tutulduğunu ve sıralamanın doğruluğunu test eder."""
    config = load_config("config.yaml")

    df = pd.DataFrame([
        {"film_index": 0, "title": "Film A", "display_title": "Film A (2000)", "year": 2000, "genres": "Drama", "overview": "Drama story about life."},
        {"film_index": 1, "title": "Film B", "display_title": "Film B (2005)", "year": 2005, "genres": "Drama", "overview": "Another drama story about people."},
        {"film_index": 2, "title": "Film C", "display_title": "Film C (2010)", "year": 2010, "genres": "Action", "overview": "Fast action explosions and racing."},
    ])

    # Sahte 3 boyutlu normalize embedding'ler
    # Film A ile Film B vektörleri birbirine çok yakın (yüksek benzerlik)
    # Film C ise zıt/uzak
    v_a = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    v_b = np.array([0.85, 0.15, 0.0], dtype=np.float32)
    v_c = np.array([0.0, 0.1, 0.99], dtype=np.float32)

    # L2 normalize
    v_a /= np.linalg.norm(v_a)
    v_b /= np.linalg.norm(v_b)
    v_c /= np.linalg.norm(v_c)

    embeddings = np.vstack([v_a, v_b, v_c])

    # Film A (index 0) için öneri arıyoruz
    results = search_by_movie(
        selected_film_index=0,
        df=df,
        embeddings=embeddings,
        config=config,
        override_top_k=2
    )

    # 1) Film A kendisi sonuçlarda YER ALMAMALIDIR!
    result_indices = [r["film_index"] for r in results]
    assert 0 not in result_indices

    # 2) En benzer film olarak Film B (index 1) ilk sırada çıkmalıdır
    assert results[0]["film_index"] == 1
    assert results[0]["score_pct"] > results[1]["score_pct"]
