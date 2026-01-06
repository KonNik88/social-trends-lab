import argparse
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from datasets import load_dataset

# --- HF HUB: force higher HTTP timeout globally ---
def _configure_hf_http_timeout(timeout_s: int = 120):
    """
    HuggingFace Hub sometimes uses its own default timeout=10 inside requests.
    This hook forces a larger timeout via huggingface_hub HTTP backend.
    """
    try:
        from huggingface_hub import configure_http_backend
        import requests

        def backend_factory():
            s = requests.Session()
            # increase retries at HTTP layer too
            adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=3)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            # store default timeout in session for our custom send wrapper
            s.request = _wrap_request_with_timeout(s.request, timeout_s)
            return s

        configure_http_backend(backend_factory=backend_factory)
        print(f"[INFO] huggingface_hub HTTP backend configured: timeout={timeout_s}s")
    except Exception as e:
        print(f"[WARN] Could not configure huggingface_hub http backend: {type(e).__name__}: {e}")


def _wrap_request_with_timeout(orig_request, timeout_s: int):
    def wrapped(method, url, **kwargs):
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = timeout_s
        return orig_request(method, url, **kwargs)
    return wrapped


def pick_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    body = (row.get("selftext") or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def to_canonical(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    # created_at
    created_at = pd.NaT
    if "created_utc" in df.columns:
        s = pd.to_numeric(df["created_utc"], errors="coerce")
        created_at = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    elif "created_at" in df.columns:
        created_at = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    # post_id
    if "id" in df.columns:
        post_id = df["id"].astype(str)
    elif "post_id" in df.columns:
        post_id = df["post_id"].astype(str)
    else:
        post_id = pd.Series([f"{source_name}_{i}" for i in range(len(df))])

    # url
    if "permalink" in df.columns:
        url = df["permalink"].astype(str).where(df["permalink"].notna(), None)
        url = url.apply(lambda x: f"https://www.reddit.com{x}" if isinstance(x, str) and x.startswith("/") else x)
    elif "url" in df.columns:
        url = df["url"].astype(str).where(df["url"].notna(), None)
    else:
        url = pd.Series([None] * len(df))

    author = df["author"].astype(str) if "author" in df.columns else pd.Series([None] * len(df))
    subreddit = df["subreddit"].astype(str) if "subreddit" in df.columns else pd.Series([None] * len(df))

    # text
    if {"title", "selftext"}.intersection(df.columns):
        text = df.apply(lambda r: pick_text(r.to_dict()), axis=1)
    elif "text" in df.columns:
        text = df["text"].fillna("").astype(str)
    else:
        text = pd.Series([""] * len(df))

    out = pd.DataFrame(
        {
            "post_id": post_id,
            "created_at": created_at,
            "source": source_name,
            "lang": None,
            "text": text.fillna("").astype(str),
            "url": url,
            "author": author,
            "subreddit": subreddit,
        }
    )
    out = out[out["text"].str.len() > 0].copy()
    return out


def parse_splits(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def load_split_with_retries(
    dataset: str,
    split: str,
    streaming: bool,
    retries: int,
    base_sleep_s: int,
):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(dataset, split=split, streaming=streaming)
        except Exception as e:
            last_err = e
            sleep_s = base_sleep_s * attempt
            print(f"[WARN] load_dataset failed (attempt {attempt}/{retries}) split={split} streaming={streaming}: "
                  f"{type(e).__name__}: {e}")
            print(f"[WARN] sleeping {sleep_s}s …")
            time.sleep(sleep_s)
    raise last_err


def save_partial(out_path: Path, df: pd.DataFrame):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[INFO] partial saved: {out_path.name} | rows={len(df):,}")


def main():
    ap = argparse.ArgumentParser(description="Download HF Reddit dataset by splits into per-split parquet shards.")
    ap.add_argument("--dataset", default="HuggingFaceGECLM/REDDIT_submissions")
    ap.add_argument("--splits", required=True, help="Comma-separated splits, e.g. technology,science,books")
    ap.add_argument("--limit", type=int, default=1_000_000, help="Total rows across all splits")
    ap.add_argument("--out_dir", default="data/raw/hf_splits_1m", help="Output directory for shards")
    ap.add_argument("--retries", type=int, default=12)
    ap.add_argument("--sleep", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing shard files")
    ap.add_argument("--no_streaming", action="store_true", help="Force non-streaming mode (slower, sometimes more stable)")
    ap.add_argument("--save_every", type=int, default=10_000, help="Save partial file every N rows per split")
    args = ap.parse_args()

    # env hints for HF
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ["HF_HUB_HTTP_TIMEOUT"] = str(args.timeout)

    # force timeout in huggingface_hub http backend
    _configure_hf_http_timeout(args.timeout)

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_list = parse_splits(args.splits)
    per_split = max(1, args.limit // len(split_list))

    print("=" * 100)
    print(f"[INFO] dataset: {args.dataset}")
    print(f"[INFO] splits ({len(split_list)}): {split_list}")
    print(f"[INFO] total limit: {args.limit:,} => per split: {per_split:,}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] timeout={args.timeout}s retries={args.retries} base_sleep={args.sleep}s")
    print("=" * 100)

    total_rows = 0

    for sp in split_list:
        out_path = out_dir / f"reddit_{sp}_{per_split}.parquet"

        if out_path.exists() and not args.overwrite:
            n_existing = len(pd.read_parquet(out_path))
            print(f"[INFO] shard exists -> skip: {out_path.name} | rows={n_existing:,}")
            total_rows += n_existing
            continue

        print(f"[INFO] downloading split={sp} target={per_split:,}")

        # choose mode
        streaming = False if args.no_streaming else True

        # try streaming first, fallback to non-streaming if needed
        try:
            ds = load_split_with_retries(args.dataset, sp, streaming=streaming, retries=args.retries, base_sleep_s=args.sleep)
        except Exception as e_stream:
            if streaming:
                print(f"[WARN] streaming failed for split={sp}. Fallback to non-streaming. Last error: {e_stream}")
                ds = load_split_with_retries(args.dataset, sp, streaming=False, retries=args.retries, base_sleep_s=args.sleep)
            else:
                raise

        rows = []
        last_save_at = 0

        # iterate dataset
        for i, row in enumerate(ds):
            rows.append(row)

            # periodic partial save
            if args.save_every > 0 and (i + 1) - last_save_at >= args.save_every:
                df_part = pd.DataFrame(rows)
                canon_part = to_canonical(df_part, source_name=f"hf:{args.dataset}:{sp}")
                canon_part = canon_part.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
                save_partial(out_path, canon_part)
                last_save_at = i + 1

            if i + 1 >= per_split:
                break

        df = pd.DataFrame(rows)
        canon = to_canonical(df, source_name=f"hf:{args.dataset}:{sp}")
        canon = canon.drop_duplicates(subset=["post_id"]).reset_index(drop=True)

        canon.to_parquet(out_path, index=False)
        total_rows += len(canon)

        print(f"[INFO] saved: {out_path.name} | rows={len(canon):,}")
        print("-" * 100)

    print("=" * 100)
    print(f"[DONE] shards written (sum rows): {total_rows:,}")
    print(f"[DONE] folder: {out_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
