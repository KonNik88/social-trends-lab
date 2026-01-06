import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# add src/ to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import Cfg, ProjectPaths, load_yaml
from utils.logging import get_logger

logger = get_logger()


def ensure_dirs(paths: ProjectPaths):
    (paths.root / "artifacts").mkdir(exist_ok=True)
    (paths.root / "artifacts/embeddings").mkdir(parents=True, exist_ok=True)
    (paths.root / "artifacts/bertopic_model").mkdir(parents=True, exist_ok=True)
    (paths.root / "artifacts/reports").mkdir(parents=True, exist_ok=True)


def clean_stage(paths: ProjectPaths, cfg: Cfg) -> Path:
    raw_path = paths.abs(cfg.get("data.raw_path"))
    out_path = paths.abs(cfg.get("data.clean_path", "artifacts/posts_clean.parquet"))

    logger.info(f"Load RAW: {raw_path}")
    df = pd.read_parquet(raw_path)

    min_len = int(cfg.get("preprocess.min_text_len", 20))
    max_chars = int(cfg.get("preprocess.max_chars", 2000))

    df["text"] = df["text"].fillna("").astype(str)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() >= min_len].copy()
    df["text"] = df["text"].str.slice(0, max_chars)

    if bool(cfg.get("preprocess.dedup.enabled", True)):
        df["text_norm"] = df["text"].str.lower()
        df = df.drop_duplicates(subset=["text_norm"]).copy()
        df = df.drop(columns=["text_norm"], errors="ignore")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved CLEAN: {out_path} | rows={len(df):,}")
    return out_path


def embeddings_stage(paths: ProjectPaths, cfg: Cfg, clean_path: Path) -> Path:
    from sentence_transformers import SentenceTransformer
    import torch

    df = pd.read_parquet(clean_path)
    texts = df["text"].fillna("").astype(str).tolist()

    model_name = cfg.get("embeddings.model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(cfg.get("embeddings.batch_size", 256))
    out_path = paths.abs(cfg.get("embeddings.out_path", "artifacts/embeddings/sbert.npy"))
    dtype = str(cfg.get("embeddings.dtype", "float16")).lower()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Embeddings: model={model_name} device={device} batch={batch_size}")

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    emb = emb.astype(np.float16) if dtype == "float16" else emb.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    logger.info(f"Saved EMB: {out_path} | shape={emb.shape} dtype={emb.dtype}")
    return out_path


def topics_stage(paths: ProjectPaths, cfg: Cfg, clean_path: Path, emb_path: Path) -> tuple[Path, Path, Path]:
    from bertopic import BERTopic
    from umap import UMAP
    import hdbscan

    df = pd.read_parquet(clean_path)
    emb = np.load(emb_path)

    n = len(df)
    max_fit_docs = int(cfg.get("topics.max_fit_docs", 200_000))
    fit_n = min(max_fit_docs, n)

    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=fit_n, replace=False)
    idx.sort()

    docs_fit = df.iloc[idx]["text"].tolist()
    emb_fit = emb[idx]

    umap_cfg = cfg.get("topics.umap", {}) or {}
    hdb_cfg = cfg.get("topics.hdbscan", {}) or {}

    umap_model = UMAP(
        n_neighbors=int(umap_cfg.get("n_neighbors", 30)),
        n_components=int(umap_cfg.get("n_components", 5)),
        min_dist=float(umap_cfg.get("min_dist", 0.0)),
        metric=str(umap_cfg.get("metric", "cosine")),
        random_state=42,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=int(hdb_cfg.get("min_cluster_size", 50)),
        min_samples=hdb_cfg.get("min_samples", None),
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True,
    )

    logger.info(f"Fitting BERTopic on subset: {fit_n:,}/{n:,}")
    topic_model.fit_transform(docs_fit, embeddings=emb_fit)

    logger.info("Transforming ALL docs to assign topics…")
    topics_all, probs_all = topic_model.transform(df["text"].tolist(), embeddings=emb)

    if probs_all is not None and len(probs_all) == n:
        topic_prob = probs_all.max(axis=1)
    else:
        topic_prob = np.full(n, np.nan)

    topics_df = pd.DataFrame(
        {
            "post_id": df["post_id"].astype(str),
            "topic_id": topics_all,
            "topic_prob": topic_prob,
            "created_at": df.get("created_at"),
            "url": df.get("url"),
        }
    )

    info = topic_model.get_topic_info()

    def topic_keywords(tid: int) -> str:
        if tid == -1:
            return ""
        ws = topic_model.get_topic(tid) or []
        return ", ".join([w for w, _ in ws[:10]])

    info["keywords"] = info["Topic"].apply(topic_keywords)

    out_topics_path = paths.abs(cfg.get("topics.out_topics_path", "artifacts/topics.parquet"))
    out_info_path = paths.abs(cfg.get("topics.out_info_path", "artifacts/topics_info.parquet"))
    model_dir = paths.abs(cfg.get("topics.model_dir", "artifacts/bertopic_model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "bertopic.pkl"

    out_topics_path.parent.mkdir(parents=True, exist_ok=True)
    out_info_path.parent.mkdir(parents=True, exist_ok=True)

    topics_df.to_parquet(out_topics_path, index=False)
    info.to_parquet(out_info_path, index=False)

    topic_model.save(model_path)

    logger.info(f"Saved TOPICS: {out_topics_path} | rows={len(topics_df):,}")
    logger.info(f"Saved TOPIC INFO: {out_info_path} | rows={len(info):,}")
    logger.info(f"Saved MODEL: {model_path}")

    return out_topics_path, out_info_path, model_path


def trends_stage(paths: ProjectPaths, cfg: Cfg, topics_path: Path) -> Path:
    df = pd.read_parquet(topics_path)

    created = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.assign(created_at=created).dropna(subset=["created_at"]).copy()

    freq = str(cfg.get("trends.freq", "W"))
    out_path = paths.abs(cfg.get("trends.out_path", "artifacts/trends.parquet"))

    df["time_bin"] = df["created_at"].dt.to_period(freq).dt.start_time

    agg = (
        df.groupby(["time_bin", "topic_id"], as_index=False)
        .agg(n_posts=("post_id", "count"), avg_prob=("topic_prob", "mean"))
    )

    totals = agg.groupby("time_bin")["n_posts"].transform("sum")
    agg["share"] = agg["n_posts"] / totals

    agg = agg.sort_values(["topic_id", "time_bin"]).copy()
    agg["n_posts_prev"] = agg.groupby("topic_id")["n_posts"].shift(1)
    agg["growth"] = (agg["n_posts"] - agg["n_posts_prev"]) / agg["n_posts_prev"]
    agg["growth"] = agg["growth"].replace([np.inf, -np.inf], np.nan)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)
    logger.info(f"Saved TRENDS: {out_path} | rows={len(agg):,}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--skip_topics", action="store_true")
    ap.add_argument("--skip_embeddings", action="store_true")
    ap.add_argument("--skip_trends", action="store_true")
    args = ap.parse_args()

    paths = ProjectPaths.detect()
    cfg = Cfg(load_yaml(paths.abs(args.config)))
    ensure_dirs(paths)

    clean_path = clean_stage(paths, cfg)

    # embeddings
    if args.skip_embeddings:
        emb_path = paths.abs(cfg.get("embeddings.out_path"))
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found (skip_embeddings=True): {emb_path}")
        logger.info(f"Skip embeddings. Using: {emb_path}")
    else:
        emb_path = embeddings_stage(paths, cfg, clean_path)

    # topics
    topics_path = None
    if args.skip_topics:
        topics_path = paths.abs(cfg.get("topics.out_topics_path"))
        if not topics_path.exists():
            logger.info("Skip topics requested, but topics file does not exist yet.")
            logger.info(f"Expected: {topics_path}")
            logger.info("=> Skipping trends as well (nothing to aggregate).")
            args.skip_trends = True
        else:
            logger.info(f"Skip topics. Using: {topics_path}")
    else:
        topics_path, _, _ = topics_stage(paths, cfg, clean_path, emb_path)

    # trends
    if args.skip_trends:
        logger.info("Skip trends.")
    else:
        if topics_path is None or not Path(topics_path).exists():
            raise FileNotFoundError("Cannot compute trends: topics file is missing.")
        trends_stage(paths, cfg, topics_path)

    logger.info("Pipeline done.")


if __name__ == "__main__":
    main()