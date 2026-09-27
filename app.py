"""Film Benzerliği Öneri Sistemi - Streamlit Web Arayüzü (app.py).

Gelişmiş koyu tema, glassmorphism estetiği ve modern kart tasarımıyla
kullanıcıların film adı veya konu metniyle semantik öneriler almasını sağlar.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from src.config import load_config
from src.embeddings import get_device
from src.search import (
    fuzzy_find_movies,
    encode_query,
    search_recommendations,
    search_by_movie,
)

# -----------------------------------------------------------------------------
# Sayfa Yapılandırması
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Film Benzerliği Öneri Motoru",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Özel CSS - Modern Koyu Tema ve Glassmorphism Estetiği
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Ana arka plan rengi */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(18, 24, 38, 1) 0%, rgba(9, 13, 21, 1) 90%);
    color: #F3F4F6;
}

/* Üst Başlık Banner */
.hero-container {
    padding: 2rem 1.5rem 1.5rem 1.5rem;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(236, 72, 153, 0.1) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    backdrop-filter: blur(12px);
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818CF8, #C084FC, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #9CA3AF;
}

/* Film Kartı Stili (Glassmorphism) */
.movie-card {
    background: rgba(26, 32, 44, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(16px);
    transition: all 0.25s ease-in-out;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
.movie-card:hover {
    border-color: rgba(129, 140, 248, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
}

.movie-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.movie-title {
    font-size: 1.28rem;
    font-weight: 600;
    color: #FFFFFF;
}
.movie-year {
    color: #9CA3AF;
    font-size: 1rem;
    font-weight: 400;
}

.score-badge {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: #FFFFFF;
    font-size: 0.95rem;
    font-weight: 700;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    box-shadow: 0 2px 10px rgba(79, 70, 229, 0.4);
}

.genre-pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.07);
    color: #E0E7FF;
    font-size: 0.8rem;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    margin-right: 0.4rem;
    margin-bottom: 0.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.overview-text {
    font-size: 0.94rem;
    color: #D1D5DB;
    line-height: 1.55;
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
}

.why-similar-box {
    background: rgba(99, 102, 241, 0.1);
    border-left: 3px solid #818CF8;
    padding: 0.5rem 0.8rem;
    border-radius: 4px 8px 8px 4px;
    font-size: 0.87rem;
    color: #C7D2FE;
}

.disclaimer-text {
    font-size: 0.85rem;
    color: #6B7280;
    text-align: center;
    margin-top: 1.5rem;
    padding: 0.8rem;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Veri ve Model Önbelleği (st.cache_resource)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Veri ve embedding önbelleği yükleniyor...")
def load_system_resources():
    """Uygulama yapılandırmasını, temiz veriyi ve embedding matrisini yükler."""
    config = load_config("config.yaml")

    data_path = Path(config.paths.clean_data_file)
    emb_path = Path(config.paths.embeddings_file)

    if not data_path.is_file() or not emb_path.is_file():
        return config, None, None, False

    df = pd.read_parquet(data_path)
    embeddings = np.load(emb_path)
    return config, df, embeddings, True


@st.cache_resource(show_spinner="Semantik model hafızaya alınıyor...")
def get_cached_model(model_name: str, device: str):
    """Sentence-Transformers modelini tek seferde yükleyip hafızada tutar."""
    from sentence_transformers import SentenceTransformer
    dev = get_device(device)
    print(f"[Streamlit] Model yükleniyor: {model_name} (Cihaz: {dev})")
    model = SentenceTransformer(model_name, device=dev)
    return model


# -----------------------------------------------------------------------------
# Kaynakları Yükle
# -----------------------------------------------------------------------------
config, df, embeddings, is_ready = load_system_resources()

# -----------------------------------------------------------------------------
# Başlık ve Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎬 Semantik Film Öneri Motoru</div>
    <div class="hero-subtitle">
        Yapay zeka embedding vektörleri ile derin anlamsal benzerlikleri keşfedin.
        İster bir film adı seçin, ister aklınızdaki hikayeyi Türkçe veya İngilizce anlatın.
    </div>
</div>
""", unsafe_allow_html=True)

# Önbellek henüz oluşturulmadıysa kullanıcıya yol göster
if not is_ready:
    st.error("⚠️ Sistem önbelleği henüz hazırlanmamış!")
    st.info(
        "Lütfen terminal üzerinden şu komutu bir kez çalıştırarak verileri ve embedding'leri hazırlayın:\n\n"
        "```bash\npython prepare.py\n```"
    )
    st.stop()

# -----------------------------------------------------------------------------
# Yan Panel (Filtreler ve Ayarlar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Arama & Filtre Ayarları")

    # 1) Öneri Sayısı (top_k)
    top_k = st.slider(
        "Öneri Sayısı (top_k)",
        min_value=1,
        max_value=20,
        value=config.top_k,
        help="Listelenecek maksimum benzer film adedi."
    )

    # 2) Asgari Benzerlik Eşiği
    min_sim = st.slider(
        "Asgari Benzerlik Eşiği",
        min_value=0.0,
        max_value=0.9,
        value=float(config.min_similarity),
        step=0.05,
        help="Bu eşiğin altında kalan filmler filtrelenir."
    )

    st.markdown("---")
    st.subheader("🎯 Aday Havuzu Filtreleri")

    # 3) Tür Filtresi (Tüm benzersiz türleri topla)
    all_genres = set()
    for g_str in df["genres"].dropna():
        for g in g_str.split(","):
            cleaned_g = g.strip()
            if cleaned_g and cleaned_g != "Bilinmiyor":
                all_genres.add(cleaned_g)
    sorted_genres = sorted(list(all_genres))

    selected_genres = st.multiselect(
        "Film Türü",
        options=sorted_genres,
        default=[],
        help="Yalnızca seçilen türleri içeren filmleri listeler."
    )

    # 4) Yıl Aralığı Filtresi
    valid_years = df["year"].dropna().astype(int)
    min_dataset_year = int(valid_years.min()) if len(valid_years) > 0 else 1920
    max_dataset_year = int(valid_years.max()) if len(valid_years) > 0 else 2026

    year_range = st.slider(
        "Yapım Yılı Aralığı",
        min_value=min_dataset_year,
        max_value=max_dataset_year,
        value=(config.filters.default_min_year, config.filters.default_max_year),
        step=1
    )

    st.markdown("---")
    st.subheader("🔬 İleri Düzey Ayarlar")

    score_display_mode = st.radio(
        "Skor Gösterim Biçimi",
        options=["raw_cosine", "minmax"],
        index=0 if config.score_display == "raw_cosine" else 1,
        format_func=lambda x: "Doğrudan Cosine (0-100%)" if x == "raw_cosine" else "Grupta Min-Max Ölçekleme"
    )
    # Seçilen modu konfigürasyona yansıt
    config.score_display = score_display_mode

    enable_hybrid = st.checkbox(
        "Hibrit Arama (TF-IDF Destekli)",
        value=config.hybrid.enabled,
        help="Embedding benzerliği ile TF-IDF kelime benzerliğini harmanlar."
    )
    tfidf_weight = config.hybrid.tfidf_weight
    if enable_hybrid:
        tfidf_weight = st.slider("TF-IDF Ağırlığı", 0.05, 0.50, tfidf_weight, 0.05)

    st.caption(f"Veri Seti: {len(df):,} film | Model: {config.model_name}")

# -----------------------------------------------------------------------------
# Ana Sekmeler: 1) Film Adıyla Ara, 2) Konu Anlatarak Ara
# -----------------------------------------------------------------------------
tab_title, tab_text = st.tabs(["🎬 Film Adıyla Ara", "📝 Konu Anlatarak Ara"])

results = None
search_context = ""

# --- SEKME 1: FİLM ADIYLA ARA ---
with tab_title:
    st.markdown("##### Benzerini bulmak istediğiniz filmi arayın:")

    search_input = st.text_input(
        "Film adı yazın (Yazım hataları otomatik düzeltilir):",
        placeholder="Örn: Godfahter, Inception, Interstellar, Se7en...",
        key="film_search_input"
    )

    selected_movie_idx = None
    if search_input.strip():
        # Fuzzy arama ile en yakın 10 filmi listele
        candidate_matches = fuzzy_find_movies(search_input, df, limit=10, score_cutoff=35.0)

        if candidate_matches:
            options_dict = {
                f"{m['display_title']}  —  {m['genres']} (Eşleşme: %{m['fuzzy_score']})": m["film_index"]
                for m in candidate_matches
            }
            selected_label = st.selectbox(
                "En yakın eşleşen filmlerden birini seçin:",
                options=list(options_dict.keys()),
                key="movie_dropdown"
            )
            selected_movie_idx = options_dict[selected_label]
        else:
            st.warning("Bu isimle eşleşen bir film bulunamadı. Lütfen farklı bir yazım deneyin.")

    col1, col2 = st.columns([1, 4])
    with col1:
        search_by_movie_btn = st.button("🚀 Benzer Filmleri Getir", key="btn_by_movie", type="primary")

    if search_by_movie_btn and selected_movie_idx is not None:
        chosen_row = df.iloc[selected_movie_idx]
        search_context = f"'{chosen_row['display_title']}' Filmine Benzerler"
        with st.spinner("Önbellekteki embedding'lerle anlamsal benzerlik hesaplanıyor..."):
            results = search_by_movie(
                selected_film_index=selected_movie_idx,
                df=df,
                embeddings=embeddings,
                config=config,
                filter_genre=selected_genres if selected_genres else None,
                filter_min_year=year_range[0],
                filter_max_year=year_range[1],
                override_top_k=top_k,
                override_min_similarity=min_sim
            )

# --- SEKME 2: KONU ANLATARAK ARA ---
with tab_text:
    st.markdown("##### Aklınızdaki senaryoyu, konuyu veya anahtar olayları anlatın:")

    sample_prompts = [
        "A detective investigates a series of mysterious murders while trying to uncover a conspiracy.",
        "Bir dedektif, bir komployu ortaya çıkarmaya çalışırken gizemli cinayetleri araştırır.",
        "A group of astronauts travel through a wormhole in space in search of a new habitable planet for humanity.",
        "A mind-bending heist inside the subconscious mind where dreams exist within dreams."
    ]

    selected_sample = st.selectbox(
        "Örnek senaryo şablonları:",
        ["Örnek seçin veya aşağıya kendiniz yazın..."] + sample_prompts,
        index=0
    )

    initial_text = selected_sample if selected_sample != "Örnek seçin veya aşağıya kendiniz yazın..." else ""
    query_text_input = st.text_area(
        "Senaryo / Konu Metni (Türkçe veya İngilizce yazabilirsiniz):",
        value=initial_text,
        placeholder="Örn: İki kardeş çocukluklarında ayrılır ve yıllar sonra karşıt taraflarda karşılaşırlar...",
        height=130,
        key="topic_text_area"
    )

    col3, col4 = st.columns([1, 4])
    with col3:
        search_by_text_btn = st.button("🔍 Semantik Eşleşmeleri Bul", key="btn_by_text", type="primary")

    if search_by_text_btn and query_text_input.strip():
        search_context = f"Konu: '{query_text_input[:60]}...'" if len(query_text_input) > 60 else f"Konu: '{query_text_input}'"
        with st.spinner("Sorgu vektörleştiriliyor ve benzerlikler aranıyor..."):
            model = get_cached_model(config.model_name, config.device)
            q_vector = encode_query(query_text_input, model, max_words=config.max_seq_length // 2)

            results = search_recommendations(
                query_vector=q_vector,
                df=df,
                embeddings=embeddings,
                config=config,
                query_text_for_explain=query_text_input,
                filter_genre=selected_genres if selected_genres else None,
                filter_min_year=year_range[0],
                filter_max_year=year_range[1],
                override_top_k=top_k,
                override_min_similarity=min_sim,
                enable_hybrid=enable_hybrid,
                tfidf_weight=tfidf_weight
            )

# -----------------------------------------------------------------------------
# Sonuçların Gösterimi
# -----------------------------------------------------------------------------
if results is not None:
    st.markdown("---")
    st.subheader(f"📊 Öneri Sonuçları: {search_context}")

    if len(results) == 0:
        st.info("Kriterlerinize uygun benzer film bulunamadı. Lütfen filtreleri gevşetmeyi veya asgari benzerlik eşiğini düşürmeyi deneyin.")
    else:
        for rank, r in enumerate(results, 1):
            score = r["score_pct"]
            year_str = f"({int(r['year'])})" if pd.notna(r["year"]) else ""

            # Tür etiketleri
            genre_pills_html = "".join(
                f'<span class="genre-pill">{g.strip()}</span>'
                for g in r["genres"].split(",") if g.strip()
            )

            # "Neden benzer?" kutusu
            exp = r.get("explanation", {})
            why_html = ""
            if exp.get("keywords"):
                kws = ", ".join(exp["keywords"])
                why_html = f'<div class="why-similar-box">💡 <strong>Ortak Anahtar Kavramlar:</strong> {kws}</div>'
            elif exp.get("message"):
                why_html = f'<div class="why-similar-box">💡 {exp["message"]}</div>'

            card_html = f"""
            <div class="movie-card">
                <div class="movie-header">
                    <div>
                        <span class="movie-title">{rank}. {r['title']}</span>
                        <span class="movie-year">{year_str}</span>
                    </div>
                    <div class="score-badge">%{score:.1f} Benzerlik</div>
                </div>
                <div>{genre_pills_html}</div>
                <div class="overview-text">{r['overview']}</div>
                {why_html}
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        st.markdown(
            '<div class="disclaimer-text">ℹ️ <em>Not: Bu yüzde, gerçek bir olasılık değil, modele göre göreli benzerlik skorudur.</em></div>',
            unsafe_allow_html=True
        )
