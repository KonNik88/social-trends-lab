from __future__ import annotations

import argparse
import html as ihtml
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from sentence_transformers import SentenceTransformer

from bertopic import BERTopic
from umap import UMAP
import hdbscan

# Representation models
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DELETED_RE = re.compile(r"^\s*\[(deleted|removed)\]\s*$", re.IGNORECASE)

PROMO_RE_DEFAULT = re.compile(
    r"\b(buy|discount|promo|promotion|free\s+shipping|sale|deal|coupon|"
    r"amazon|amzn\.to|t\.co|bit\.ly|tinyurl|affiliate|referral|subscribe|"
    r"limited\s+time|offer)\b",
    re.IGNORECASE,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_ws(s: str) -> str:
    return " ".join(str(s).split())


def html_unescape(s: str) -> str:
    return ihtml.unescape(str(s))


def to_float16_if_requested(x: np.ndarray, dtype: str) -> np.ndarray:
    d = str(dtype).lower()
    if d in {"float16", "fp16"}:
        return x.astype(np.float16, copy=False)
    if d in {"float32", "fp32"}:
        return x.astype(np.float32, copy=False)
    return x


def maybe_langdetect(texts: pd.Series) -> Optional[pd.Series]:
    try:
        from langdetect import detect  # type: ignore
    except Exception:
        return None

    def _det(x: str) -> str:
        x = x.strip()
        if len(x) < 40:
            return "unknown"
        try:
            return detect(x)
        except Exception:
            return "unknown"

    return texts.fillna("").astype(str).map(_det)


def is_link_only(text: str, min_non_url_chars: int = 40, max_url_ratio: float = 0.7) -> bool:
    t = text.strip()
    if not t:
        return True

    urls = URL_RE.findall(t)
    if not urls:
        return False

    t_no_urls = URL_RE.sub(" ", t)
    t_no_urls = normalize_ws(t_no_urls)
    non_url_len = len(t_no_urls)

    url_chars = sum(len(u) for u in urls)
    total_len = max(len(t), 1)
    url_ratio = url_chars / total_len

    if non_url_len < min_non_url_chars:
        return True
    if url_ratio > max_url_ratio:
        return True
    return False


def clean_stage(cfg: Dict[str, Any]) -> pd.DataFrame:
    raw_path = Path(cfg["data"]["raw_path"])
    clean_path = Path(cfg["data"]["clean_path"])

    preprocess = cfg.get("preprocess", {})
    text_col = preprocess.get("text_col", "text")
    title_col = preprocess.get("title_col", None)

    min_text_len = int(preprocess.get("min_text_len", 20))
    max_chars = int(preprocess.get("max_chars", 2000))
    dedup_enabled = bool(preprocess.get("dedup", {}).get("enabled", True))

    v2 = preprocess.get("v2", {})
    do_html_unescape = bool(v2.get("html_unescape", True))
    drop_deleted_removed = bool(v2.get("drop_deleted_removed", True))
    drop_link_only = bool(v2.get("drop_link_only", True))
    link_only_min_non_url_chars = int(v2.get("link_only_min_non_url_chars", 40))
    link_only_max_url_ratio = float(v2.get("link_only_max_url_ratio", 0.7))

    drop_promo = bool(v2.get("drop_promo", True))
    promo_regex = v2.get("promo_regex", None)
    promo_re = re.compile(promo_regex, re.IGNORECASE) if promo_regex else PROMO_RE_DEFAULT

    lang_filter = v2.get("lang_filter", {})
    lang_enabled = bool(lang_filter.get("enabled", False))
    lang_keep = str(lang_filter.get("keep", "en")).lower()
    lang_min_chars = int(lang_filter.get("min_chars", 80))

    print(f"[INFO] Load RAW: {raw_path.resolve()}")
    df = pd.read_parquet(raw_path)
    print(f"[INFO] RAW loaded: rows={len(df):,} cols={len(df.columns)}")

    if text_col not in df.columns:
        raise RuntimeError(f"RAW parquet must contain '{text_col}' column. Found: {list(df.columns)}")

    if title_col and title_col in df.columns:
        df[text_col] = (
            df[title_col].fillna("").astype(str).str.strip()
            + "\n\n"
            + df[text_col].fillna("").astype(str).str.strip()
        ).str.strip()
    else:
        df[text_col] = df[text_col].fillna("").astype(str)

    if do_html_unescape:
        df[text_col] = df[text_col].map(html_unescape)
    df[text_col] = df[text_col].map(normalize_ws)

    before = len(df)
    df = df[df[text_col].str.len() >= min_text_len].copy()
    df[text_col] = df[text_col].str.slice(0, max_chars)
    print(f"[INFO] Filter len>= {min_text_len}: {before:,} -> {len(df):,}")
    before = len(df)

    if drop_deleted_removed:
        df = df[~df[text_col].str.match(DELETED_RE)].copy()
        print(f"[INFO] Drop [deleted]/[removed]: {before:,} -> {len(df):,}")
        before = len(df)

    if drop_link_only:
        mask_link_only = df[text_col].map(
            lambda x: is_link_only(
                x,
                min_non_url_chars=link_only_min_non_url_chars,
                max_url_ratio=link_only_max_url_ratio,
            )
        )
        df = df[~mask_link_only].copy()
        print(f"[INFO] Drop link-only: {before:,} -> {len(df):,}")
        before = len(df)

    if drop_promo:
        df = df[~df[text_col].str.contains(promo_re)].copy()
        print(f"[INFO] Drop promo-patterns: {before:,} -> {len(df):,}")
        before = len(df)

    if lang_enabled:
        long_mask = df[text_col].str.len() >= lang_min_chars
        n_long = int(long_mask.sum())
        print(f"[INFO] Lang filter enabled: detecting on {n_long:,} texts (len>={lang_min_chars})...")
        langs = pd.Series(["unknown"] * len(df), index=df.index)
        detected = maybe_langdetect(df.loc[long_mask, text_col])
        if detected is None:
            print("[WARN] langdetect not installed -> skipping language filter")
        else:
            langs.loc[long_mask] = detected
            df = df[langs.isin([lang_keep, "unknown"])].copy()
            print(f"[INFO] Keep lang={lang_keep} or unknown: {before:,} -> {len(df):,}")
            before = len(df)

    if dedup_enabled:
        before = len(df)
        key = df[text_col].str.lower()
        df = df.loc[~key.duplicated()].copy()
        print(f"[INFO] Dedup by text: {before:,} -> {len(df):,}")

    ensure_dir(clean_path)
    df.to_parquet(clean_path, index=False)
    print(f"[INFO] Saved CLEAN (v2): {clean_path.resolve()} | rows={len(df):,}")
    return df


def embeddings_stage(cfg: Dict[str, Any], clean_df: pd.DataFrame) -> np.ndarray:
    emb_cfg = cfg["embeddings"]
    out_path = Path(emb_cfg["out_path"])
    text_col = cfg.get("preprocess", {}).get("text_col", "text")

    batch_size = int(emb_cfg.get("batch_size", 256))
    dtype = str(emb_cfg.get("dtype", "float16"))
    model_name = str(emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    normalize_embeddings = bool(emb_cfg.get("normalize_embeddings", True))

    texts = clean_df[text_col].fillna("").astype(str).tolist()

    print(f"[INFO] Embeddings: model={model_name} | batch={batch_size} | normalize={normalize_embeddings}")
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    emb = to_float16_if_requested(emb, dtype)

    ensure_dir(out_path)
    np.save(out_path, emb)
    print(f"[INFO] Saved EMB: {out_path.resolve()} | shape={emb.shape} | dtype={emb.dtype}")
    return emb


def topics_stage(cfg: Dict[str, Any], clean_df: pd.DataFrame, embeddings: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tcfg = cfg["topics"]
    text_col = cfg.get("preprocess", {}).get("text_col", "text")

    out_topics_path = Path(tcfg["out_topics_path"])
    out_info_path = Path(tcfg["out_info_path"])
    model_dir = Path(tcfg["model_dir"])

    max_fit_docs = int(tcfg.get("max_fit_docs", 200000))
    seed = int(tcfg.get("seed", 42))

    nr_topics = tcfg.get("nr_topics", None)

    um = tcfg.get("umap", {})
    hd = tcfg.get("hdbscan", {})

    umap_model = UMAP(
        n_neighbors=int(um.get("n_neighbors", 10)),
        n_components=int(um.get("n_components", 5)),
        min_dist=float(um.get("min_dist", 0.05)),
        metric=str(um.get("metric", "cosine")),
        random_state=seed,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=int(hd.get("min_cluster_size", 120)),
        min_samples=int(hd.get("min_samples", 5)),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    rep_cfg = tcfg.get("representation", {})
    rep_enabled = bool(rep_cfg.get("enabled", True))
    rep_diversity = float(rep_cfg.get("mmr_diversity", 0.3))

    representation_model = None
    embedding_model = None

    # KeyBERTInspired requires topic_model.embedding_model to embed representative docs.
    # Even if we pass embeddings to fit/transform, it still needs embedding_model for KeyBERT step.
    if rep_enabled:
        representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(diversity=rep_diversity),
        }
        model_name = str(cfg["embeddings"].get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        embedding_model = SentenceTransformer(model_name)

    n = len(clean_df)
    fit_n = min(max_fit_docs, n)
    print(f"[INFO] Fitting BERTopic on subset: {fit_n:,}/{n:,}")

    rs = np.random.RandomState(seed)
    idx = rs.choice(n, size=fit_n, replace=False)
    idx.sort()

    emb_fit = embeddings[idx]
    docs_fit = clean_df.iloc[idx][text_col].fillna("").astype(str).tolist()

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        verbose=True,
        nr_topics=nr_topics,
    )

    topic_model.fit(docs_fit, embeddings=emb_fit)

    print("[INFO] Transforming ALL docs to assign topics")
    docs_all = clean_df[text_col].fillna("").astype(str).tolist()
    topics, probs = topic_model.transform(docs_all, embeddings=embeddings)

    out_df = clean_df.copy()
    out_df["topic_id"] = topics
    out_df["topic_prob"] = np.nan if probs is None else probs.max(axis=1)

    keep_cols = []
    for c in ["post_id", "created_at", "url", text_col, "topic_id", "topic_prob"]:
        if c in out_df.columns:
            keep_cols.append(c)
    out_df = out_df[keep_cols].copy()

    topic_info = topic_model.get_topic_info()

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "bertopic.pkl"
    topic_model.save(str(model_path), serialization="pickle")

    ensure_dir(out_topics_path)
    ensure_dir(out_info_path)
    out_df.to_parquet(out_topics_path, index=False)
    topic_info.to_parquet(out_info_path, index=False)

    print(f"[INFO] Saved TOPICS: {out_topics_path.resolve()} | rows={len(out_df):,}")
    print(f"[INFO] Saved TOPIC INFO: {out_info_path.resolve()} | rows={len(topic_info):,}")
    print(f"[INFO] Saved MODEL: {model_path.resolve()}")

    return out_df, topic_info


def trends_stage(cfg: Dict[str, Any], topics_df: pd.DataFrame) -> pd.DataFrame:
    trcfg = cfg["trends"]
    out_path = Path(trcfg["out_path"])
    freq = str(trcfg.get("freq", "W"))

    if "created_at" not in topics_df.columns:
        raise RuntimeError("topics_df must have created_at for trends computation")

    df = topics_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at"]).copy()

    df["time_bin"] = df["created_at"].dt.to_period(freq).dt.to_timestamp()

    g = df.groupby(["time_bin", "topic_id"], as_index=False).agg(
        n_posts=("topic_id", "size"),
        avg_prob=("topic_prob", "mean"),
    )

    totals = g.groupby("time_bin", as_index=False)["n_posts"].sum().rename(columns={"n_posts": "n_total"})
    g = g.merge(totals, on="time_bin", how="left")
    g["share"] = g["n_posts"] / g["n_total"]

    g = g.sort_values(["topic_id", "time_bin"]).copy()
    g["n_posts_prev"] = g.groupby("topic_id")["n_posts"].shift(1)
    g["growth"] = (g["n_posts"] - g["n_posts_prev"]) / g["n_posts_prev"]
    g["growth"] = g["growth"].replace([np.inf, -np.inf], np.nan)

    ensure_dir(out_path)
    g.to_parquet(out_path, index=False)
    print(f"[INFO] Saved TRENDS: {out_path.resolve()} | rows={len(g):,}")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to yaml config")
    ap.add_argument("--skip_embeddings", action="store_true")
    ap.add_argument("--skip_topics", action="store_true")
    ap.add_argument("--skip_trends", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    clean_path = Path(cfg["data"]["clean_path"])
    if args.skip_embeddings and clean_path.exists():
        print(f"[INFO] Using existing CLEAN: {clean_path.resolve()}")
        clean_df = pd.read_parquet(clean_path)
    else:
        clean_df = clean_stage(cfg)

    emb_path = Path(cfg["embeddings"]["out_path"])
    if args.skip_embeddings and emb_path.exists():
        print(f"[INFO] Skip embeddings. Using: {emb_path.resolve()}")
        embeddings = np.load(emb_path)
    else:
        embeddings = embeddings_stage(cfg, clean_df=clean_df)

    if len(clean_df) != embeddings.shape[0]:
        raise RuntimeError(
            f"clean rows != embeddings rows: clean={len(clean_df):,} emb={embeddings.shape[0]:,}. "
            f"Recompute embeddings or reuse existing clean."
        )

    if args.skip_topics:
        topics_path = Path(cfg["topics"]["out_topics_path"])
        info_path = Path(cfg["topics"]["out_info_path"])
        print(f"[INFO] Skip topics. Using: {topics_path.resolve()}")
        topics_df = pd.read_parquet(topics_path)
        _ = pd.read_parquet(info_path)
    else:
        topics_df, _ = topics_stage(cfg, clean_df=clean_df, embeddings=embeddings)

    if not args.skip_trends:
        trends_stage(cfg, topics_df)

    print("[INFO] Pipeline v2 done.")


if __name__ == "__main__":
    main()
