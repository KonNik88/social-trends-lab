from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_TOPICS = Path("artifacts/topics.parquet")
DEFAULT_INFO = Path("artifacts/topics_info.parquet")
DEFAULT_REPORT = Path("artifacts/reports/sanity_topics_report.txt")
DEFAULT_EXAMPLES = Path("artifacts/reports/topic_examples.parquet")


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_str(x, max_len: int = 220) -> str:
    s = "" if x is None else str(x)
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_parquet(path)


def _infer_topic_info_columns(info: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "topic": _pick_col(info, ["topic_id", "Topic", "topic"]),
        "count": _pick_col(info, ["Count", "count", "topic_count", "size"]),
        "name": _pick_col(info, ["Name", "name", "topic_name", "label"]),
        "repr": _pick_col(info, ["Representation", "representation", "keywords", "top_words"]),
    }


def _infer_topics_columns(topics: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "topic": _pick_col(topics, ["topic_id", "Topic", "topic"]),
        "prob": _pick_col(topics, ["topic_prob", "probability", "prob", "score"]),
        "text": _pick_col(topics, ["text", "body", "content", "document", "post_text"]),
        "title": _pick_col(topics, ["title", "post_title"]),
        "created": _pick_col(topics, ["created_at", "created_utc", "timestamp", "date", "datetime"]),
        "url": _pick_col(topics, ["url", "permalink"]),
    }


def _compute_basic_stats(topics_df: pd.DataFrame, tcol: str, noise_topic: int = -1) -> Dict[str, float]:
    n_docs = len(topics_df)
    n_unique = int(topics_df[tcol].nunique(dropna=False))
    noise_share = float((topics_df[tcol] == noise_topic).mean()) if n_docs else 0.0

    non_noise = topics_df.loc[topics_df[tcol] != noise_topic, tcol]
    n_topics_wo_noise = int(non_noise.nunique()) if len(non_noise) else 0

    vc = topics_df[tcol].value_counts(dropna=False)
    top20 = vc.head(20)

    return {
        "n_docs": float(n_docs),
        "n_unique_topics_including_noise": float(n_unique),
        "n_topics_excluding_noise": float(n_topics_wo_noise),
        "noise_share": float(noise_share),
        "largest_topic_size": float(top20.iloc[0]) if len(top20) else 0.0,
    }


def _build_examples(
    topics_df: pd.DataFrame,
    tcol: str,
    text_col: Optional[str],
    title_col: Optional[str],
    created_col: Optional[str],
    url_col: Optional[str],
    prob_col: Optional[str],
    per_topic: int = 3,
    max_topics: int = 30,
    seed: int = 42,
    noise_topic: int = -1,
    include_noise: bool = False,
) -> pd.DataFrame:
    rng = random.Random(seed)
    df = topics_df.copy()

    # display text
    if title_col and text_col and title_col in df.columns and text_col in df.columns:
        df["_display_text"] = (
            df[title_col].fillna("").astype(str).str.strip()
            + "\n\n"
            + df[text_col].fillna("").astype(str).str.strip()
        ).str.strip()
        display_col = "_display_text"
    elif text_col and text_col in df.columns:
        display_col = text_col
    elif title_col and title_col in df.columns:
        display_col = title_col
    else:
        df["_display_text"] = ""
        display_col = "_display_text"

    vc = df[tcol].value_counts()
    topic_ids = vc.index.tolist()
    if not include_noise:
        topic_ids = [tid for tid in topic_ids if tid != noise_topic]
    topic_ids = topic_ids[:max_topics]

    rows = []
    for tid in topic_ids:
        sub = df[df[tcol] == tid]
        if sub.empty:
            continue

        if prob_col and prob_col in sub.columns:
            sub2 = sub.sort_values(prob_col, ascending=False).head(per_topic * 5)
            idxs = sub2.index.tolist()
            pick = idxs[:per_topic] if len(idxs) >= per_topic else idxs
        else:
            idxs = sub.index.tolist()
            pick = rng.sample(idxs, k=min(per_topic, len(idxs)))

        for i, ix in enumerate(pick, start=1):
            r = df.loc[ix]
            rows.append(
                {
                    "topic_id": int(tid) if pd.notna(tid) else tid,
                    "rank": i,
                    "topic_prob": float(r[prob_col]) if (prob_col and prob_col in df.columns and pd.notna(r[prob_col])) else np.nan,
                    "created_at": r[created_col] if (created_col and created_col in df.columns) else None,
                    "url": r[url_col] if (url_col and url_col in df.columns) else None,
                    "text": _safe_str(r[display_col]),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity-check BERTopic topics artifacts (report-only).")
    ap.add_argument("--topics", type=str, default=str(DEFAULT_TOPICS))
    ap.add_argument("--info", type=str, default=str(DEFAULT_INFO))
    ap.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    ap.add_argument("--examples", type=str, default=str(DEFAULT_EXAMPLES))
    ap.add_argument("--noise_topic", type=int, default=-1)
    ap.add_argument("--examples_per_topic", type=int, default=3)
    ap.add_argument("--max_topics", type=int, default=30)
    ap.add_argument("--include_noise_in_examples", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    topics_path = Path(args.topics)
    info_path = Path(args.info)
    report_path = Path(args.report)
    examples_path = Path(args.examples)

    topics_df = _load_parquet(topics_path)
    info_df = _load_parquet(info_path)

    tcols = _infer_topics_columns(topics_df)
    icols = _infer_topic_info_columns(info_df)

    if not tcols["topic"]:
        raise RuntimeError(f"Can't find topic_id column in {topics_path}. Columns: {list(topics_df.columns)}")
    tcol = tcols["topic"]

    stats = _compute_basic_stats(topics_df, tcol=tcol, noise_topic=args.noise_topic)

    vc = topics_df[tcol].value_counts(dropna=False)
    top20 = vc.head(20)

    # info lookup
    info_map = {}
    info_topic_col = icols["topic"]
    if info_topic_col and info_topic_col in info_df.columns:
        tmp = info_df.rename(columns={info_topic_col: "_tid"}).copy()
        name_col = icols["name"]
        repr_col = icols["repr"]
        count_col = icols["count"]
        for _, r in tmp.iterrows():
            tid = r.get("_tid")
            if pd.isna(tid):
                continue
            info_map[int(tid)] = {
                "name": r.get(name_col) if name_col else None,
                "repr": r.get(repr_col) if repr_col else None,
                "count": r.get(count_col) if count_col else None,
            }

    ex_df = _build_examples(
        topics_df=topics_df,
        tcol=tcol,
        text_col=tcols["text"],
        title_col=tcols["title"],
        created_col=tcols["created"],
        url_col=tcols["url"],
        prob_col=tcols["prob"],
        per_topic=args.examples_per_topic,
        max_topics=args.max_topics,
        seed=args.seed,
        noise_topic=args.noise_topic,
        include_noise=args.include_noise_in_examples,
    )

    _ensure_dir(report_path)
    _ensure_dir(examples_path)
    ex_df.to_parquet(examples_path, index=False)

    lines = []
    lines.append("=== SANITY TOPICS REPORT ===")
    lines.append(f"topics_path: {topics_path.resolve()}")
    lines.append(f"info_path:   {info_path.resolve()}")
    lines.append("")
    lines.append(f"docs: {int(stats['n_docs']):,}")
    lines.append(f"n_unique (including noise): {int(stats['n_unique_topics_including_noise']):,}")
    lines.append(f"n_topics (excluding noise): {int(stats['n_topics_excluding_noise']):,}")
    lines.append(f"noise_topic: {args.noise_topic}")
    lines.append(f"noise_share: {stats['noise_share']:.3f}")
    lines.append("")
    lines.append("--- Top-20 topic sizes ---")
    for tid, cnt in top20.items():
        tid_int = int(tid) if pd.notna(tid) and str(tid).lstrip("-").isdigit() else tid
        meta = info_map.get(int(tid_int), {}) if isinstance(tid_int, int) else {}
        name = _safe_str(meta.get("name", ""), 120) if meta else ""
        repr_ = _safe_str(meta.get("repr", ""), 160) if meta else ""
        lines.append(f"topic={tid_int:>4} | size={int(cnt):>7} | name={name} | repr={repr_}")

    lines.append("")
    lines.append("--- Examples saved ---")
    lines.append(f"examples_path: {examples_path.resolve()}")
    lines.append(f"examples_rows: {len(ex_df):,}")
    lines.append("")
    lines.append("--- Columns detected ---")
    lines.append(json.dumps({"topics_cols": tcols, "info_cols": icols}, indent=2))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Report saved: {report_path}")
    print(f"[OK] Examples saved: {examples_path}")


if __name__ == "__main__":
    main()
