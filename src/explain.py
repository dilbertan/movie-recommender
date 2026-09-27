"""Benzerlik açıklama modülü (TF-IDF tabanlı 'Neden Benzer?' analizi).

Sorgu metni ile önerilen film özeti arasındaki ortak kavramları bulur.
Sorgu Türkçe ise çok dilli modelin anlamsal eşleme yaptığını nazikçe belirtir.
"""

from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from src.utils import is_turkish_text


def explain_similarity(
    query_text: str,
    movie_overview: str,
    max_keywords: int = 5
) -> Dict[str, Any]:
    """Sorgu metni ile önerilen filmin özeti arasındaki benzerliği açıklar.

    Args:
        query_text: Kullanıcının girdiği konu/özet sorgusu veya seçilen filmin özeti.
        movie_overview: Önerilen filmin İngilizce özeti.
        max_keywords: Döndürülecek en fazla ortak anahtar kelime sayısı.

    Returns:
        dict: {
            'keywords': list[str],
            'explanation_type': 'keywords' | 'turkish_skipped' | 'semantic',
            'message': str | None
        }
    """
    # 1) Türkçe sorgu kontrolü
    if is_turkish_text(query_text):
        return {
            "keywords": [],
            "explanation_type": "turkish_skipped",
            "message": "Sorgu Türkçe olduğu için ortak kelime analizi atlandı. (Model çok dilli semantik eşleştirme kullandı.)"
        }

    # 2) İngilizce metinler için TF-IDF analizi
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"  # En az 3 harfli kelimeler
        )
        tfidf_matrix = vectorizer.fit_transform([query_text, movie_overview])
        feature_names = np.array(vectorizer.get_feature_names_out())

        vec_query = tfidf_matrix[0].toarray().flatten()
        vec_movie = tfidf_matrix[1].toarray().flatten()

        # Her iki metinde de geçen kelimelerin ortak önem skoru (çarpımı)
        common_importance = vec_query * vec_movie
        nonzero_indices = np.where(common_importance > 0)[0]

        if len(nonzero_indices) == 0:
            return {
                "keywords": [],
                "explanation_type": "semantic",
                "message": "Doğrudan ortak kelime yerine anlamsal kavram benzerliği bulundu."
            }

        # En yüksek ağırlığa göre sırala
        sorted_indices = nonzero_indices[np.argsort(-common_importance[nonzero_indices])]
        top_words = list(feature_names[sorted_indices[:max_keywords]])

        return {
            "keywords": top_words,
            "explanation_type": "keywords",
            "message": f"Ortak anahtar kavramlar: {', '.join(top_words)}"
        }
    except Exception:
        return {
            "keywords": [],
            "explanation_type": "semantic",
            "message": "Anlamsal bağlam eşleşmesi."
        }
