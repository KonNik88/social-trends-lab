import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["post_id", "created_at", "source", "lang", "text", "url", "author", "subreddit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory with parquet shards")
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--dedup", action="store_true", help="Drop duplicates by post_id (recommended)")
    ap.add_argument("--sort", action="store_true", help="Sort by created_at after merge")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    in_dir = project_root / args.in_dir
    out_path = project_root / args.out

    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in: {in_dir}")

    print("=" * 90)
    print(f"[INFO] Merge from: {in_dir}")
    print(f"[INFO] Files: {len(files)}")
    print(f"[INFO] Out: {out_path}")
    print("=" * 90)

    dfs = []
    total = 0
    for fp in files:
        df = pd.read_parquet(fp)
        total += len(df)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"File {fp.name} missing columns: {missing}")

        # keep column order stable
        df = df[REQUIRED_COLS].copy()
        dfs.append(df)
        print(f"[INFO] Read: {fp.name} | rows={len(df):,}")

    out = pd.concat(dfs, ignore_index=True)
    print("-" * 90)
    print(f"[INFO] Concatenated rows: {len(out):,} (sum shards: {total:,})")

    if args.dedup:
        before = len(out)
        out = out.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
        print(f"[INFO] Dedup by post_id: {before:,} -> {len(out):,}")

    # normalize datetime
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce")

    if args.sort:
        out = out.sort_values("created_at").reset_index(drop=True)
        print("[INFO] Sorted by created_at")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print("=" * 90)
    print(f"[DONE] Saved merged parquet: {out_path} | rows={len(out):,}")
    print("=" * 90)


if __name__ == "__main__":
    main()
