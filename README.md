# Semantic Movie Recommendation Engine

A local, open-source semantic recommendation system for movies. It allows discovering relevant titles using natural language plot queries, thematic descriptions, or existing movie titles with typo tolerance.

The project provides both an interactive **Streamlit web interface** (`app.py`) and a scriptable **Command-Line Interface** (`cli.py`).

---

## Features

- **Semantic Vector Search**: Uses the multilingual `BAAI/bge-m3` model to match conceptual meaning and themes rather than strict keyword hits.
- **Cross-Lingual Querying**: Supports querying in multiple languages (e.g., Turkish or English) against an English catalog.
- **Fuzzy Title Resolution**: Powered by `rapidfuzz` to gracefully handle misspellings (e.g., `"Godfahter"` resolves to `"The Godfather"`).
- **Candidate Filtering**: Applies genre and release-year filters directly during candidate retrieval to ensure the target `top_k` count is satisfied.
- **Query Movie Exclusion**: Automatically excludes the source title from results when searching by movie.
- **Similarity Explanation**: Identifies shared thematic keywords using TF-IDF or reports cross-lingual semantic alignment.
- **Disk Caching**: Precomputed embeddings are persisted to `cache/embeddings.npy` for sub-second query latency after initial setup.
- **Hardware Acceleration**: Automatic CUDA detection with fp16 execution and seamless CPU fallback.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/dilbertan/movie-recommender.git
cd movie-recommender
```

### 2. Virtual Environment Setup
Python 3.10 or higher is recommended.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for NVIDIA GPU users:**  
> To utilize CUDA acceleration, install the appropriate PyTorch build:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Usage

### 1. Data Preparation & Embedding Generation
Run `prepare.py` once before starting the application:

```bash
python prepare.py
```
This downloads top-rated TMDB movie records from Hugging Face, cleans the text, saves `data/movies_clean.parquet`, and writes dense vector representations to `cache/embeddings.npy`. Subsequent runs will detect the existing cache and load immediately.

---

### 2. Web Interface (Streamlit)
Start the local web application:

```bash
streamlit run app.py
```
Access the UI at `http://localhost:8501`.

**Interface capabilities:**
- Search by movie title with autocomplete and typo correction.
- Search by free-form natural language plot descriptions.
- Sidebar controls for candidate count (`top_k`), minimum similarity threshold, genre filtering, release year range, scoring mode (`raw_cosine` or `minmax`), and optional TF-IDF hybrid weighting.

---

### 3. Command Line Interface (CLI)

Use `cli.py` for headless queries and automation:

**Search by title (typo-tolerant):**
```bash
python cli.py --title "Godfahter" --top-k 3
```

**Search by plot description:**
```bash
python cli.py --text "A detective investigates a series of mysterious murders while trying to uncover a conspiracy."
```

**Cross-lingual query (Turkish):**
```bash
python cli.py --text "Bir dedektif, bir komployu ortaya çıkarmaya çalışırken gizemli cinayetleri araştırır." --top-k 5
```

**Filtered search:**
```bash
python cli.py --text "Space exploration and time travel" --genre "Science Fiction" --min-year 2000
```

---

## Configuration (`config.yaml`)

Core parameters can be adjusted via `config.yaml`:

```yaml
# Number of recommendations returned
top_k: 3

# Minimum similarity score threshold (0.0 - 1.0)
min_similarity: 0.0

# Score display mode:
# - raw_cosine: Direct cosine similarity mapped to 0-100% (Default)
# - minmax: Relative min-max scaling across candidate batch
score_display: "raw_cosine"

# Sentence-Transformers model
model_name: "BAAI/bge-m3"

# Runtime device ('auto', 'cuda', 'cpu')
device: "auto"

# Batch size for embedding generation
batch_size: 16

# Maximum sequence length
max_seq_length: 512

dataset:
  name: "jason1966/ahsanaseer_top-rated-tmdb-movies-10k"
  max_films: 10000
  min_year: 1900
  max_words: 300
  random_seed: 42

hybrid:
  enabled: false       # Enable TF-IDF hybrid scoring
  tfidf_weight: 0.2    # Weight of TF-IDF in total similarity score
```

---

## Architecture & Methodology

Standard lexical keyword retrieval (such as SQL `LIKE` or basic inverted indexes) often fails when queries use synonyms or abstract descriptions.

This engine employs a dense semantic vector space pipeline:

1. **Embedding Generation**: The `BAAI/bge-m3` encoder converts plot summaries into 1024-dimensional normalized vectors representing semantic content.
2. **Similarity Computation**: Dot products between query vectors and L2-normalized corpus vectors produce exact cosine similarity scores.
3. **Optimized Inference**: Stored NumPy matrices enable vectorized matrix-vector multiplications, returning ranked candidates in milliseconds.

---

## Testing

Test suites can be executed with `pytest`:

```bash
pytest -v
```

**Coverage overview:**
- `tests/test_config.py`: Configuration parsing, constraints, and defaults.
- `tests/test_data.py`: Preprocessing, text truncation, duplicate resolution, and schema validation.
- `tests/test_search.py`: Similarity computation, fuzzy title lookup, source movie exclusion, and chunking.
- `tests/test_explain.py`: Keyword extraction via TF-IDF and cross-lingual heuristics.

---

## Troubleshooting

- **Cache missing**: If prompted with a missing cache error, run `python prepare.py` to rebuild vector files.
- **CUDA memory constraints**: If running into GPU memory limitations, reduce `batch_size` (e.g., to `8` or `4`) in `config.yaml` or set `device: "cpu"`.
- **Terminal character encoding**: Windows terminals default to legacy code pages. The application invokes UTF-8 standard output configurations automatically; running `chcp 65001` manually in PowerShell also resolves encoding artifacts.

---

## Roadmap

- Metadata enrichment including directors, writers, and primary cast members.
- Vector store integration (Chroma, Qdrant, or FAISS) for larger catalog scalability.
- Media card view and TMDB poster asset resolution.

---

## License

Distributed under the [MIT License](LICENSE).
