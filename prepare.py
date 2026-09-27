"""Veri hazırlık ve önbellek oluşturma komut dosyası.

Kullanım:
    python prepare.py
    python prepare.py --force   (Mevcut verileri ve önbelleği sıfırlayıp baştan üretir)
"""

import argparse
import sys
import time
from pathlib import Path

# Windows konsol kodlama uyumluluğu (cp1254 / UnicodeEncodeError önlemi)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import load_config, ConfigError
from src.data import get_or_create_clean_data
from src.embeddings import compute_and_cache_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Film Benzerliği Öneri Sistemi - Veri ve Embedding Hazırlık Aracı"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Mevcut temizlenmiş veriyi ve önbellek embedding'lerini yok sayarak sıfırdan üretir."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Kullanılacak yapılandırma dosyasının yolu (Varsayılan: config.yaml)."
    )
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 70)
    print("[FILM BENZERLIGI ONERI SISTEMI] - HAZIRLIK ASAMASI")
    print("=" * 70)

    try:
        config = load_config(args.config)
    except ConfigError as err:
        print(f"HATA: Yapılandırma okunamadı: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Yapılandırma yüklendi: Model={config.model_name}, Donanım={config.device}")

    # 1. Adım: Veri çekme ve temizleme
    print("\n[1/2] Film verisi kontrol ediliyor...")
    try:
        df = get_or_create_clean_data(config, force_reload=args.force)
        print(f"[OK] Kullanıma hazır film sayısı: {len(df):,}")
    except Exception as exc:
        print(f"HATA: Veri hazırlığı başarısız oldu: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Adım: Embedding üretimi ve önbellekleme
    print("\n[2/2] Vektör embedding'leri ve önbellek kontrol ediliyor...")
    try:
        embeddings = compute_and_cache_embeddings(df, config, force_recompute=args.force)
        print(f"[OK] Embedding matrisi hazır: {embeddings.shape[0]:,} film, {embeddings.shape[1]} vektör boyutu.")
    except Exception as exc:
        print(f"HATA: Embedding üretimi başarısız oldu: {exc}", file=sys.stderr)
        sys.exit(1)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[TAMAMLANDI] HAZIRLIK BASARIYLA TAMAMLANDI! (Toplam süre: {total_time:.2f} saniye)")
    print(f"Temizlenmiş Veri: {config.paths.clean_data_file}")
    print(f"Embedding Önbelleği: {config.paths.embeddings_file}")
    print("Artık 'streamlit run app.py' veya 'python cli.py' komutlarını kullanabilirsiniz.")
    print("=" * 70)


if __name__ == "__main__":
    main()
