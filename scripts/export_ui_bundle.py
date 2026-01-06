from __future__ import annotations

import argparse
import html as ihtml
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


DEFAULT_TOPICS = Path("artifacts/topics.parquet")
DEFAULT_INFO = Path("artifacts/topics_info.parquet")
DEFAULT_TRENDS = Path("artifacts/trends.parquet")
DEFAULT_UI_DIR = Path("artifacts/ui")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_parquet(path)


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_topics_df(df: pd.DataFrame) -> pd.DataFrame:
    # Required: post_id, topic_id, topic_prob, created_at, url (+ text optional)
    col_map = {}

    c_post = _pick_col(df, ["post_id", "id"])
    c_topic = _pick_col(df, ["topic_id", "Topic", "topic"])
    c_prob = _pick_col(df, ["topic_prob", "probability", "prob", "score"])
    c_created = _pick_col(df, ["created_at", "created_utc", "timestamp", "date", "datetime"])
    c_url = _pick_col(df, ["url", "permalink"])

    if not c_topic:
        raise RuntimeError(f"Can't find topic_id column in topics df. Columns: {list(df.columns)}")
    if not c_prob:
        # allow missing: fill NaN
        df["topic_prob"] = np.nan
        c_prob = "topic_prob"

    if c_post and c_post != "post_id":
        col_map[c_post] = "post_id"
    if c_topic != "topic_id":
        col_map[c_topic] = "topic_id"
    if c_prob != "topic_prob":
        col_map[c_prob] = "topic_prob"
    if c_created and c_created != "created_at":
        col_map[c_created] = "created_at"
    if c_url and c_url != "url":
        col_map[c_url] = "url"

    out = df.rename(columns=col_map).copy()

    # Ensure core cols exist
    if "post_id" not in out.columns:
        out["post_id"] = [f"row_{i}" for i in range(len(out))]
    if "created_at" in out.columns:
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")
    else:
        out["created_at"] = pd.NaT
    if "url" not in out.columns:
        out["url"] = None

    out["topic_id"] = pd.to_numeric(out["topic_id"], errors="coerce").fillna(-1).astype(int)
    out["topic_prob"] = pd.to_numeric(out["topic_prob"], errors="coerce")

    # Keep stable column order (text optional)
    keep = ["post_id", "topic_id", "topic_prob", "created_at", "url"]
    if "text" in out.columns:
        keep.append("text")
    out = out[[c for c in keep if c in out.columns]].copy()
    return out


def _normalize_info_df(df: pd.DataFrame) -> pd.DataFrame:
    # Expect BERTopic-style: Topic, Count, Name, Representation, ...
    out = df.copy()
    if "Topic" in out.columns and "topic_id" not in out.columns:
        out = out.rename(columns={"Topic": "topic_id"})
    if "topic_id" not in out.columns:
        # try alternative
        c = _pick_col(out, ["topic_id", "topic"])
        if c and c != "topic_id":
            out = out.rename(columns={c: "topic_id"})
    if "topic_id" not in out.columns:
        raise RuntimeError(f"Can't find topic_id in topics_info. Columns: {list(out.columns)}")

    if "Name" not in out.columns:
        out["Name"] = out.get("Representation", "")
    if "keywords" not in out.columns:
        # if Representation looks like list/str, keep as is
        out["keywords"] = out.get("Representation", "")

    out["topic_id"] = pd.to_numeric(out["topic_id"], errors="coerce").fillna(-1).astype(int)
    return out


def _normalize_trends_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time_bin" not in out.columns:
        c = _pick_col(out, ["time_bin", "created_at", "date", "datetime"])
        if c and c != "time_bin":
            out = out.rename(columns={c: "time_bin"})
    if "time_bin" not in out.columns:
        raise RuntimeError(f"Can't find time_bin in trends. Columns: {list(out.columns)}")

    if "topic_id" not in out.columns:
        c = _pick_col(out, ["topic_id", "Topic", "topic"])
        if c and c != "topic_id":
            out = out.rename(columns={c: "topic_id"})
    if "topic_id" not in out.columns:
        raise RuntimeError(f"Can't find topic_id in trends. Columns: {list(out.columns)}")

    # Ensure numeric + datetime
    out["time_bin"] = pd.to_datetime(out["time_bin"], utc=True, errors="coerce")
    out["topic_id"] = pd.to_numeric(out["topic_id"], errors="coerce").fillna(-1).astype(int)

    # Ensure required numeric cols exist (UI uses these)
    for c in ["n_posts", "avg_prob", "share", "growth"]:
        if c not in out.columns:
            out[c] = np.nan

    # Fix types
    for c in ["n_posts"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["avg_prob", "share", "growth"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Stable order
    keep = ["time_bin", "topic_id", "n_posts", "avg_prob", "share", "n_posts_prev", "growth"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()
    return out


def _safe_preview_text(x: str, max_len: int = 500) -> str:
    s = "" if x is None else str(x)
    s = ihtml.unescape(s)
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _build_topic_examples(
    topics_full: pd.DataFrame,
    max_topics: int,
    per_topic: int,
    min_prob: float,
    include_noise: bool,
    noise_topic: int,
    seed: int,
) -> pd.DataFrame:
    df = topics_full.copy()

    # Need text for examples
    if "text" not in df.columns:
        # if we don't have text, we cannot create examples
        return pd.DataFrame(columns=["topic_id", "topic_prob", "created_at", "url", "text"])

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    # choose topics by size
    vc = df["topic_id"].value_counts(dropna=False)
    topic_ids = vc.index.tolist()
    if not include_noise:
        topic_ids = [t for t in topic_ids if int(t) != noise_topic]
    topic_ids = topic_ids[:max_topics]

    rng = np.random.default_rng(seed)

    rows = []
    for tid in topic_ids:
        sub = df[df["topic_id"] == int(tid)].copy()
        if sub.empty:
            continue

        # filter by prob
        if "topic_prob" in sub.columns:
            sub = sub[sub["topic_prob"].fillna(-1) >= float(min_prob)].copy()

        if sub.empty:
            continue

        # prefer highest prob; if probs are NaN, random sample
        if sub["topic_prob"].notna().any():
            sub = sub.sort_values("topic_prob", ascending=False).head(per_topic * 10)
            pick = sub.head(per_topic)
        else:
            idx = sub.index.to_numpy()
            k = min(per_topic, len(idx))
            pick = sub.loc[rng.choice(idx, size=k, replace=False)]

        for _, r in pick.iterrows():
            rows.append(
                {
                    "topic_id": int(r["topic_id"]),
                    "topic_prob": float(r["topic_prob"]) if pd.notna(r["topic_prob"]) else np.nan,
                    "created_at": r.get("created_at"),
                    "url": r.get("url"),
                    "text": _safe_preview_text(r.get("text")),
                }
            )

    ex = pd.DataFrame(rows)
    if not ex.empty:
        ex = ex.sort_values(["topic_id", "topic_prob"], ascending=[True, False]).reset_index(drop=True)
    return ex


def _compute_prob_stats(df: pd.DataFrame) -> dict:
    p = pd.to_numeric(df.get("topic_prob"), errors="coerce")
    out = {
        "median": float(p.median(skipna=True)) if len(p) else None,
        "p_ge_0_2": float((p >= 0.2).mean()) if len(p) else None,
        "p_ge_0_3": float((p >= 0.3).mean()) if len(p) else None,
        "p_ge_0_4": float((p >= 0.4).mean()) if len(p) else None,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export UI bundle (topics/topics_info/trends/examples/meta) for Streamlit.")
    ap.add_argument("--topics", type=str, default=str(DEFAULT_TOPICS), help="Path to doc-level topics parquet.")
    ap.add_argument("--info", type=str, default=str(DEFAULT_INFO), help="Path to topic-level info parquet.")
    ap.add_argument("--trends", type=str, default=str(DEFAULT_TRENDS), help="Path to trends parquet.")
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_UI_DIR), help="Output directory for UI bundle.")
    ap.add_argument("--profile_name", type=str, default="default", help="Profile name written to meta.json.")
    ap.add_argument("--noise_topic", type=int, default=-1, help="Noise topic id.")

    # examples settings
    ap.add_argument("--examples_max_topics", type=int, default=60)
    ap.add_argument("--examples_per_topic", type=int, default=12)
    ap.add_argument("--examples_min_prob", type=float, default=0.30)
    ap.add_argument("--examples_include_noise", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    topics_path = Path(args.topics)
    info_path = Path(args.info)
    trends_path = Path(args.trends)
    out_dir = Path(args.out_dir)

    topics_raw = _load_parquet(topics_path)
    info_raw = _load_parquet(info_path)
    trends_raw = _load_parquet(trends_path)

    topics_df = _normalize_topics_df(topics_raw)
    info_df = _normalize_info_df(info_raw)
    trends_df = _normalize_trends_df(trends_raw)

    _ensure_dir(out_dir)

    # Stats
    n_docs = int(len(topics_df))
    n_unique = int(topics_df["topic_id"].nunique(dropna=False))
    noise_share = float((topics_df["topic_id"] == int(args.noise_topic)).mean()) if n_docs else 0.0
    prob_stats = _compute_prob_stats(topics_df)

    # Build examples (requires text column in topics parquet!)
    examples_df = _build_topic_examples(
        topics_full=topics_df,
        max_topics=int(args.examples_max_topics),
        per_topic=int(args.examples_per_topic),
        min_prob=float(args.examples_min_prob),
        include_noise=bool(args.examples_include_noise),
        noise_topic=int(args.noise_topic),
        seed=int(args.seed),
    )

    # Save bundle under standard names used by UI
    out_topics = out_dir / "topics.parquet"
    out_info = out_dir / "topics_info.parquet"
    out_trends = out_dir / "trends.parquet"
    out_examples = out_dir / "topic_examples.parquet"

    topics_df.to_parquet(out_topics, index=False)
    info_df.to_parquet(out_info, index=False)
    trends_df.to_parquet(out_trends, index=False)
    examples_df.to_parquet(out_examples, index=False)

    meta = {
        "profile_name": args.profile_name,
        "sources": {
            "topics": str(topics_path.as_posix()),
            "topics_info": str(info_path.as_posix()),
            "trends": str(trends_path.as_posix()),
        },
        "bundle": {
            "topics": str(out_topics.as_posix()),
            "topics_info": str(out_info.as_posix()),
            "trends": str(out_trends.as_posix()),
            "topic_examples": str(out_examples.as_posix()),
        },
        "stats": {
            "n_docs": n_docs,
            "n_unique_topics_including_noise": n_unique,
            "noise_topic": int(args.noise_topic),
            "noise_share": float(round(noise_share, 6)),
            "topic_prob": prob_stats,
        },
        "examples_cfg": {
            "max_topics": int(args.examples_max_topics),
            "per_topic": int(args.examples_per_topic),
            "min_prob": float(args.examples_min_prob),
            "include_noise": bool(args.examples_include_noise),
            "seed": int(args.seed),
        },
        "columns": {
            "topics_cols": list(topics_df.columns),
            "info_cols": list(info_df.columns),
            "trends_cols": list(trends_df.columns),
            "examples_cols": list(examples_df.columns),
        },
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] UI bundle exported to: {out_dir.resolve()}")
    print(f"[OK] topics         -> {out_topics}")
    print(f"[OK] topics_info    -> {out_info}")
    print(f"[OK] trends         -> {out_trends}")
    print(f"[OK] topic_examples -> {out_examples}")
    print(f"[OK] meta           -> {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
