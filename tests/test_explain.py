"""Açıklama ve TF-IDF modülü testleri."""

from src.explain import explain_similarity
from src.utils import is_turkish_text


def test_is_turkish_text():
    """Türkçe metin algılama testleri."""
    assert is_turkish_text("Bir dedektif gizemli cinayetleri araştırır.") is True
    assert is_turkish_text("Bu film çok güzel ve sürükleyici bir gerilim filmi.") is True
    assert is_turkish_text("A detective investigates mysterious murders in the city.") is False


def test_explain_similarity_turkish():
    """Türkçe sorguda ortak kelime analizinin atlanıp nazik mesaj verildiğini test eder."""
    query = "Bir dedektif gizemli olayları çözer."
    overview = "A detective investigates strange events and crimes in New York."

    result = explain_similarity(query, overview)
    assert result["explanation_type"] == "turkish_skipped"
    assert "Türkçe" in result["message"]
    assert len(result["keywords"]) == 0


def test_explain_similarity_english_keywords():
    """İngilizce sorguda ortak anahtar kelimelerin çıkarıldığını test eder."""
    query = "A detective investigates mysterious murders and a conspiracy."
    overview = "Two detectives hunt down a serial killer whose murders represent the seven deadly sins."

    result = explain_similarity(query, overview)
    assert result["explanation_type"] == "keywords"
    assert len(result["keywords"]) > 0
    # 'murders' veya 'detective' ortak kelime olmalı
    lower_kws = [k.lower() for k in result["keywords"]]
    assert any(k in lower_kws for k in ("murders", "detectives", "killer"))
