"""Yardımcı araçlar, skor normalizasyonu ve metin kontrolleri modülü."""

import re
from typing import List, Optional
import numpy as np


def normalize_score(
    raw_similarity: float,
    mode: str = "raw_cosine",
    min_in_batch: float = 0.0,
    max_in_batch: float = 1.0
) -> float:
    """Cosine benzerlik skorunu yüzde değerine (0 - 100) dönüştürür.

    Args:
        raw_similarity: -1.0 ile 1.0 arasındaki ham cosine skoru.
        mode: 'raw_cosine' veya 'minmax'.
        min_in_batch: minmax modu için gruptaki minimum skor.
        max_in_batch: minmax modu için gruptaki maksimum skor.

    Returns:
        float: 0.0 ile 100.0 arasında yuvarlanmış yüzde.
    """
    if mode == "minmax":
        spread = max_in_batch - min_in_batch
        if spread > 1e-6:
            score = (raw_similarity - min_in_batch) / spread
        else:
            score = 1.0
        return round(float(np.clip(score * 100.0, 0.0, 100.0)), 1)

    # Varsayılan: raw_cosine (negatifler 0'a kırpılır)
    clipped = max(0.0, min(1.0, float(raw_similarity)))
    return round(clipped * 100.0, 1)


# Türkçe'ye özgü karakterler ve yaygın Türkçe durak sözcükleri (stop words)
TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
TURKISH_STOPWORDS = {
    "bir", "ve", "bu", "da", "de", "ile", "için", "çok", "en", "gibi",
    "kadar", "daha", "olarak", "sonra", "ancak", "çünkü", "veya", "ya",
    "biri", "olan", "tarafından", "hakkında", "film", "filmi", "konusu",
    "adam", "kadın", "hikaye", "cinayet", "dedektif", "katil", "olay"
}


def is_turkish_text(text: str) -> bool:
    """Verilen metnin Türkçe olup olmadığını tespit eder.

    Türkçe karakterler veya yaygın Türkçe sözcükler incelenir.
    """
    if not text:
        return False

    # 1) Türkçe'ye özgü harf kontrolü
    if any(ch in TURKISH_CHARS for ch in text):
        return True

    # 2) Sözcük bazlı kontrol
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    if not words:
        return False

    turkish_word_count = sum(1 for w in words if w in TURKISH_STOPWORDS)
    # En az 2 Türkçe stopword varsa veya toplam sözcüklerin %20'sinden fazlasıysa
    return turkish_word_count >= 2 or (turkish_word_count / len(words) >= 0.2)
