# Social Trends Lab

**Taming social streams with topic modeling, trend analytics, and interactive visualizations.**

> A reproducible, GPU‑friendly pipeline that ingests social/news posts, cleans and embeds text (SBERT), discovers topics with **BERTopic**, tracks **temporal dynamics**, and serves an **interactive Streamlit dashboard**. Roadmap includes **multimodal CLIP** (text↔image) search and image clusters.

---

## Why this project
- **Portfolio‑ready**: clear problem, modern methods, clean UI, Dockerized.
- **Practical**: weekly/daily trend tracking, top posts per topic, growth/decay signals.
- **Resource‑friendly**: runs on a single GPU (e.g., RTX 2070) with MiniLM embeddings.
- **Extensible**: easy to plug new sources (Reddit/RSS/CSV), add toxicity sentiment, or go multimodal.

---

## Features (v1)
- **Ingest**: Reddit/RSS/CSV/HF datasets → normalized schema (`id, text, lang, created_at, source, url, author`).
- **Preprocess**: language filter (ru/en), cleaning, optional dedup.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (cached to disk).
- **Topics**: **BERTopic** (UMAP + HDBSCAN + c-TF‑IDF) with **time slicing** (daily/weekly).
- **Trends**: growth scores, spike detection, **top posts** by topic × time.
- **UI**: Streamlit app — topic map, timelines, faceted search, CSV export.
- **MLOps (light)**: Prefect flow for scheduled refresh; MLflow logging (coherence/diversity).

> **Planned v2**: **Multimodal CLIP** (ViT‑B/32) for text↔image search, image clusters linked to textual topics.

---

## Architecture

```
social-trends-lab/
  data/
    raw/            # raw dumps (gitignored)
    processed/
  artifacts/
    embeddings/
    bertopic_model/
    reports/
  src/
    ingest/         # reddit_api.py, rss_loader.py, hf_loader.py
    preprocess/     # clean.py, lang_filter.py, dedup.py
    embeddings/     # sbert_embed.py
    topics/         # fit_bertopic.py, dynamic_topics.py, metrics.py
    trends/         # trend_scores.py, top_posts.py, spikes.py
    ui/             # streamlit_app.py
    utils/          # io.py, config.py, logging.py, time.py
  configs/
    base.yaml
  notebooks/
    01_eda.ipynb
  docker/
    Dockerfile.ui
    docker-compose.yml
  tests/
  README.md
  .gitignore
```

**Modeling defaults**
- **Embedding**: `all-MiniLM-L6-v2` (384 dims), fp16 if available, batch 512–1024.
- **UMAP**: `n_neighbors=15–30`, `n_components=5` (tune vs stability).
- **HDBSCAN**: `min_cluster_size=30–60` (depends on N), `metric='euclidean'` on UMAP space.
- **Time slicing**: `freq='W'` initially; switch to `'D'` with more data.
- **Quality**: Topic **Coherence (c_npmi)** and **Diversity**; logged to MLflow.

---

## Quickstart

### 1) Environment
```bash
conda create -n social_trends python=3.10 -y
conda activate social_trends
pip install -r requirements.txt  # or use your existing env
```

### 2) Minimal config
`configs/base.yaml`
```yaml
data_dir: data
artifacts_dir: artifacts
language_whitelist: [ru, en]
time_freq: "W"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
umap:
  n_neighbors: 15
  n_components: 5
hdbscan:
  min_cluster_size: 40
sources:
  - kind: csv
    path: data/raw/sample_posts.csv
    text_col: text
    datetime_col: created_at
    lang_col: lang
```

### 3) Prepare a sample
Put a CSV at `data/raw/sample_posts.csv` with columns:
```
id,text,lang,created_at,source,url,author
123,"Your post text ...",en,2025-10-01,reddit,https://...,user42
```

### 4) Run pipeline (local)
```bash
python -m src.preprocess.clean --config configs/base.yaml
python -m src.embeddings.sbert_embed --config configs/base.yaml
python -m src.topics.fit_bertopic --config configs/base.yaml
python -m src.trends.trend_scores --config configs/base.yaml
```

### 5) Launch UI
```bash
streamlit run src/ui/streamlit_app.py --server.port 8501
```

---

## Streamlit UI

- **Topic Map**: 2D projection of topics (hover → keywords, click → drilldown).
- **Trends**: top rising/falling topics by period; mini-timelines.
- **Top Posts**: per topic × time, with outbound links to sources.
- **Filters**: source, language, date range, (optional) toxicity/sentiment.
- **Export**: CSV for topics/trends.

Screenshots go to `artifacts/reports/`.

---

## Reproducibility & Ops

- **Prefect**: `flows/refresh_topics.py` to schedule daily/weekly refresh.
- **MLflow**: log coherence/diversity and UMAP/HDBSCAN params per run.
- **Docker**: container for the Streamlit UI and a lightweight worker.
- **Determinism**: set seeds for UMAP/HDBSCAN where applicable (note: HDBSCAN is stochastic across embeddings).

---

## Metrics

- **Topic Coherence**: `c_npmi` on reference windows.
- **Topic Diversity**: unique top-k tokens per topic / (k × #topics).
- **Coverage**: share of docs assigned to non‑outlier topics.
- **Latency**: UI startup and interaction times.

---

## Data Sources (examples)

- **Reddit** via PRAW (subreddits of interest).
- **RSS/Atom** feeds (news/blogs/tech).
- **Hugging Face Datasets** snapshots.
- **Local CSV/Parquet** dumps.

> Please ensure you comply with each platform’s Terms of Service. This repo is for research/educational purposes.

---

## Roadmap

- [ ] Topic stability diagnostics across days/weeks.
- [ ] Lightweight **RAG**: topic summaries with LLM (optional).
- [ ] **CLIP** (ViT‑B/32) for text↔image search (v2).
- [ ] Image clusters ↔ textual topics linking (cosine in CLIP space).
- [ ] NSFW filtering for image ingestion.
- [ ] Qdrant/FAISS for fast ANN over embeddings (optional).

---

## Suggested GitHub Topics (tags)

`machine-learning`, `nlp`, `topic-modeling`, `bertopic`, `umap`, `hdbscan`, `sbert`, `sentence-transformers`, `trend-analysis`, `time-series`, `streamlit`, `data-visualization`, `mlops`, `prefect`, `mlflow`, `social-media-analytics`, `reddit`, `clustering`, `unsupervised-learning`, `docker`

---

## License

MIT — see `LICENSE`.

---

## Acknowledgements

- **BERTopic** by Maarten Grootendorst
- **Sentence-Transformers**
- **UMAP**, **HDBSCAN**
- Community datasets and open-source tools
