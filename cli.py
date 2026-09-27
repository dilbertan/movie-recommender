"""Film Benzerliği Öneri Sistemi - Komut Satırı Arayüzü (CLI).

Kullanım örnekleri:
    python cli.py --title "The Godfather"
    python cli.py --title "Godfahter" --top-k 5
    python cli.py --text "A detective investigates a series of mysterious murders."
    python cli.py --text "Bir dedektif gizemli cinayetleri araştırır." --genre "Crime"
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Windows konsol kodlama uyumluluğu (cp1254 koruması)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import load_config, ConfigError
from src.embeddings import load_model, is_cache_valid
from src.search import (
    fuzzy_find_movies,
    encode_query,
    search_recommendations,
    search_by_movie,
)


def print_results(results: list, query_desc: str, score_mode: str) -> None:
    """Arama sonuçlarını temiz ve biçimlendirilmiş bir çıktıyla terminalde gösterir."""
    print("\n" + "=" * 78)
    print(f"🎬 ARAMA SONUÇLARI: {query_desc}")
    print("=" * 78)

    if not results:
        print("\n[!] Belirtilen kriterlere uygun film bulunamadı.")
        print("İpucu: Filtreleri gevşetmeyi veya asgari benzerlik eşiğini düşürmeyi deneyin.")
        print("=" * 78 + "\n")
        return

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['display_title']}  —  %{r['score_pct']:.1f} Benzerlik")
        print(f"   Türler    : {r['genres']}")

        # Açıklama bilgisi
        exp = r.get("explanation", {})
        if exp.get("keywords"):
            kws = ", ".join(exp["keywords"])
            print(f"   Neden Benzer : [Ortak Kavramlar: {kws}]")
        elif exp.get("message"):
            print(f"   Neden Benzer : [{exp['message']}]")

        # Özet (2 satıra sararak göster)
        overview = r["overview"]
        if len(overview) > 180:
            overview = overview[:180] + "..."
        print(f"   Özet      : {overview}")
        print("   " + "-" * 72)

    print("\n[Not: Bu yüzde, gerçek bir olasılık değil, modele göre göreli benzerlik skorudur.]")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Film Benzerliği Öneri Sistemi (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--title", "-t",
        type=str,
        help="Benzerlerini bulmak istediğiniz film adı (örn. 'The Godfather' veya 'Godfahter')."
    )
    group.add_argument(
        "--text", "-q",
        type=str,
        help="Aradığınız konuyu, senaryo parçasını veya film özetini anlatan serbest metin (Türkçe veya İngilizce)."
    )

    parser.add_argument("--top-k", "-k", type=int, default=None, help="Döndürülecek öneri sayısı.")
    parser.add_argument("--genre", "-g", type=str, default=None, help="Tür filtresi (örn. 'Crime', 'Drama').")
    parser.add_argument("--min-year", type=int, default=None, help="En küçük yapım yılı filtresi.")
    parser.add_argument("--max-year", type=int, default=None, help="En büyük yapım yılı filtresi.")
    parser.add_argument("--hybrid", action="store_true", help="TF-IDF destekli hibrit aramayı etkinleştirir.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Yapılandırma dosyası.")

    args = parser.parse_args()

    # 1. Config yükle
    try:
        config = load_config(args.config)
    except ConfigError as err:
        print(f"HATA: Yapılandırma okunamadı: {err}", file=sys.stderr)
        sys.exit(1)

    # 2. Veri ve önbellek kontrolü
    clean_data_path = Path(config.paths.clean_data_file)
    emb_path = Path(config.paths.embeddings_file)

    if not clean_data_path.is_file() or not emb_path.is_file():
        print("HATA: Temizlenmiş veri veya embedding önbelleği bulunamadı!", file=sys.stderr)
        print("Lütfen önce verileri ve önbelleği hazırlamak için şu komutu çalıştırın:", file=sys.stderr)
        print("    python prepare.py\n", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_parquet(clean_data_path)
        embeddings = np.load(emb_path)
    except Exception as exc:
        print(f"HATA: Dosyalar okunurken hata oluştu: {exc}", file=sys.stderr)
        sys.exit(1)

    # 3. Arama işlemi
    if args.title:
        query_title = args.title.strip()
        # Fuzzy arama ile eşleşen en yakın filmleri bul
        matches = fuzzy_find_movies(query_title, df, limit=5, score_cutoff=40.0)

        if not matches:
            print(f"\n[!] '{query_title}' adında veya benzerinde bir film bulunamadı.")
            print("İpucu: Farklı bir yazım deneyebilir veya '--text' ile filmin konusunu yazabilirsiniz.\n")
            sys.exit(0)

        # En iyi eşleşmeyi seç
        best_match = matches[0]
        selected_idx = best_match["film_index"]
        selected_title = best_match["display_title"]

        if best_match["fuzzy_score"] < 90:
            print(f"\n[*] '{query_title}' için en yakın eşleşme seçildi: {selected_title} (Eşleşme skoru: %{best_match['fuzzy_score']})")

        results = search_by_movie(
            selected_film_index=selected_idx,
            df=df,
            embeddings=embeddings,
            config=config,
            filter_genre=args.genre,
            filter_min_year=args.min_year,
            filter_max_year=args.max_year,
            override_top_k=args.top_k
        )
        print_results(results, f"'{selected_title}' filmine benzeyenler", config.score_display)

    elif args.text:
        query_text = args.text.strip()
        if not query_text:
            print("HATA: Arama metni boş bırakılamaz.", file=sys.stderr)
            sys.exit(1)

        # Modeli yükle ve sorguyu vektörleştir
        model = load_model(config)
        query_vec = encode_query(query_text, model, max_words=config.max_seq_length // 2)

        results = search_recommendations(
            query_vector=query_vec,
            df=df,
            embeddings=embeddings,
            config=config,
            query_text_for_explain=query_text,
            filter_genre=args.genre,
            filter_min_year=args.min_year,
            filter_max_year=args.max_year,
            override_top_k=args.top_k,
            enable_hybrid=args.hybrid
        )
        short_q = query_text if len(query_text) <= 50 else query_text[:47] + "..."
        print_results(results, f"Konu: '{short_q}'", config.score_display)


if __name__ == "__main__":
    main()
