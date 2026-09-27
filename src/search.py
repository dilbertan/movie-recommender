"""Anlamsal arama, fuzzy film eşleştirme, filtreleme ve hibrit arama modülü."""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sentence_transformers import SentenceTransformer

from src.config import AppConfig
from src.utils import normalize_score
from src.explain import explain_similarity


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Çok uzun metinleri örtüşen parçalara böler.

    Args:
        text: Girdi metni.
        chunk_size: Parça başına yaklaşık kelime sayısı.
        overlap: Parçalar arasındaki örtüşen kelime sayısı.

    Returns:
        List[str]: Metin parçaları listesi.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += step
    return chunks


def encode_query(
    query_text: str,
    model: SentenceTransformer,
    max_words: int = 250
) -> np.ndarray:
    """Kullanıcı sorgusunu vektörleştirir; uzunsa parçalayıp ortalamasını alır.

    Args:
        query_text: Kullanıcının girdiği konu veya özet metni.
        model: SentenceTransformer modeli.
        max_words: Chunking eşiği kelime sayısı.

    Returns:
        np.ndarray: L2 normalize edilmiş 1D embedding vektörü.
    """
    chunks = chunk_text(query_text, chunk_size=max_words, overlap=50)

    if len(chunks) == 1:
        emb = model.encode(chunks[0], normalize_embeddings=True, convert_to_numpy=True)
    else:
        # Birden fazla parça varsa her parçanın embedding'ini al ve ortala
        chunk_embs = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
        emb = np.mean(chunk_embs, axis=0)
        # Ortalama vektörü tekrar normalize et
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

    return emb.astype(np.float32)


def fuzzy_find_movies(
    query: str,
    df: pd.DataFrame,
    limit: int = 8,
    score_cutoff: float = 30.0
) -> List[Dict[str, Any]]:
    """Film adındaki yazım hatalarını tolere ederek en yakın filmleri bulur.

    Örnek: 'Godfahter' -> 'The Godfather (1972)'
    Aynı isimli filmler yıl bilgisiyle ayırt edilir.
    """
    if not query or not query.strip():
        return []

    query_clean = query.strip()
    titles = df["display_title"].tolist()

    # Rapidfuzz ile en yakın eşleşmeleri çıkar
    matches = process.extract(
        query_clean,
        titles,
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=score_cutoff
    )

    results = []
    for display_title, score, idx in matches:
        row = df.iloc[idx]
        results.append({
            "film_index": int(row["film_index"]),
            "title": row["title"],
            "display_title": row["display_title"],
            "year": row["year"],
            "genres": row["genres"],
            "fuzzy_score": round(score, 1)
        })

    return results


def build_filter_mask(
    df: pd.DataFrame,
    genre: Optional[Union[str, List[str]]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    exclude_index: Optional[int] = None
) -> np.ndarray:
    """Aday havuzunda tür, yıl ve hariç tutulacak film filtresini uygular.

    Filtreler benzerlik hesaplandıktan sonra DEĞİL, aday havuzunda uygulanır.
    """
    mask = np.ones(len(df), dtype=bool)

    # Hariç tutulacak film (örn. sorgulanan filmin kendisi)
    if exclude_index is not None:
        mask[df["film_index"] == exclude_index] = False

    # Tür (genre) filtresi
    if genre:
        if isinstance(genre, str):
            selected_genres = [g.strip().lower() for g in genre.split(",") if g.strip()]
        else:
            selected_genres = [g.strip().lower() for g in genre if g.strip()]

        if selected_genres:
            genre_matches = df["genres"].astype(str).str.lower().apply(
                lambda val: any(target in val for target in selected_genres)
            )
            mask = mask & genre_matches.to_numpy()

    # Yıl filtresi
    if min_year is not None:
        year_gte = (df["year"].notna()) & (df["year"] >= min_year)
        mask = mask & year_gte.to_numpy()

    if max_year is not None:
        year_lte = (df["year"].notna()) & (df["year"] <= max_year)
        mask = mask & year_lte.to_numpy()

    return mask


def search_recommendations(
    query_vector: np.ndarray,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    config: AppConfig,
    query_text_for_explain: str = "",
    filter_genre: Optional[Union[str, List[str]]] = None,
    filter_min_year: Optional[int] = None,
    filter_max_year: Optional[int] = None,
    exclude_index: Optional[int] = None,
    override_top_k: Optional[int] = None,
    override_min_similarity: Optional[float] = None,
    enable_hybrid: Optional[bool] = None,
    tfidf_weight: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Normalize vektörlerle dot product yaparak en benzer filmleri sıralar.

    Aday havuzu filtreleri doğrudan uygulanır, böylece top_k tam olarak dolar.
    """
    top_k = override_top_k if override_top_k is not None else config.top_k
    min_similarity = override_min_similarity if override_min_similarity is not None else config.min_similarity
    is_hybrid = enable_hybrid if enable_hybrid is not None else config.hybrid.enabled
    tw = tfidf_weight if tfidf_weight is not None else config.hybrid.tfidf_weight

    # Aday maskesi
    mask = build_filter_mask(
        df,
        genre=filter_genre,
        min_year=filter_min_year,
        max_year=filter_max_year,
        exclude_index=exclude_index
    )

    candidate_indices = np.where(mask)[0]
    if len(candidate_indices) == 0:
        return []

    # Cosine similarity: Vektörler L2 normalize olduğu için dot product = cosine similarity
    candidate_embeddings = embeddings[candidate_indices]
    emb_sims = np.dot(candidate_embeddings, query_vector)

    # Hibrit arama (TF-IDF desteği)
    if is_hybrid and query_text_for_explain:
        try:
            candidate_texts = df.iloc[candidate_indices]["overview"].tolist()
            vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
            tfidf_mat = vectorizer.fit_transform(candidate_texts)
            q_tfidf = vectorizer.transform([query_text_for_explain])
            tfidf_sims = sklearn_cosine_similarity(tfidf_mat, q_tfidf).flatten()

            final_sims = (1.0 - tw) * emb_sims + tw * tfidf_sims
        except Exception:
            final_sims = emb_sims
    else:
        final_sims = emb_sims

    # Eşik filtresi (min_similarity)
    valid_mask = final_sims >= min_similarity
    if not np.any(valid_mask):
        return []

    candidate_indices = candidate_indices[valid_mask]
    final_sims = final_sims[valid_mask]

    # En yüksek top_k adedini seç
    k = min(top_k, len(candidate_indices))
    top_part_idx = np.argpartition(-final_sims, k - 1)[:k]
    # Seçilen top_k'yı kesin olarak sırala
    sorted_order = np.argsort(-final_sims[top_part_idx])
    chosen_indices = candidate_indices[top_part_idx[sorted_order]]
    chosen_sims = final_sims[top_part_idx[sorted_order]]

    min_sim_in_batch = float(np.min(chosen_sims)) if len(chosen_sims) > 0 else 0.0
    max_sim_in_batch = float(np.max(chosen_sims)) if len(chosen_sims) > 0 else 1.0

    results = []
    for idx, sim in zip(chosen_indices, chosen_sims):
        row = df.iloc[idx]
        pct = normalize_score(
            sim,
            mode=config.score_display,
            min_in_batch=min_sim_in_batch,
            max_in_batch=max_sim_in_batch
        )

        # "Neden benzer?" açıklaması üret
        if query_text_for_explain:
            exp = explain_similarity(query_text_for_explain, row["overview"])
        else:
            exp = {"keywords": [], "explanation_type": "semantic", "message": None}

        results.append({
            "film_index": int(row["film_index"]),
            "title": row["title"],
            "display_title": row["display_title"],
            "year": row["year"],
            "genres": row["genres"],
            "overview": row["overview"],
            "raw_similarity": round(float(sim), 4),
            "score_pct": pct,
            "explanation": exp
        })

    return results


def search_by_movie(
    selected_film_index: int,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    config: AppConfig,
    filter_genre: Optional[Union[str, List[str]]] = None,
    filter_min_year: Optional[int] = None,
    filter_max_year: Optional[int] = None,
    override_top_k: Optional[int] = None,
    override_min_similarity: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Kullanıcının seçtiği filmin hazır embedding'ini kullanarak benzer filmleri önerir.

    Sorgulanan film sonuç kümesinden kesinlikle hariç tutulur.
    """
    if selected_film_index < 0 or selected_film_index >= len(df):
        raise ValueError(f"Geçersiz film indeksi: {selected_film_index}")

    source_movie = df.iloc[selected_film_index]
    query_vector = embeddings[selected_film_index]
    query_overview = source_movie["overview"]

    return search_recommendations(
        query_vector=query_vector,
        df=df,
        embeddings=embeddings,
        config=config,
        query_text_for_explain=query_overview,
        filter_genre=filter_genre,
        filter_min_year=filter_min_year,
        filter_max_year=filter_max_year,
        exclude_index=selected_film_index,
        override_top_k=override_top_k,
        override_min_similarity=override_min_similarity
    )
