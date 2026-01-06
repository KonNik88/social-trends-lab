import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/raw/reddit_hf_mix_300k.parquet")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = root / args.path

    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    df = pd.read_parquet(path)
    print("=" * 80)
    print("RAW PATH:", path)
    print("SHAPE:", df.shape)
    print("COLUMNS:", list(df.columns))
    print("=" * 80)

    na = df.isna().mean().sort_values(ascending=False).head(15)
    print("\nTop-15 NA share:")
    print(na)

    # Quick warnings
    if "subreddit" in df.columns and df["subreddit"].isna().mean() > 0.99:
        print("\n[WARN] subreddit is almost entirely NA/None. UI фильтр по сабреддитам пока не сделать.")
    if "lang" in df.columns and df["lang"].isna().mean() > 0.99:
        print("[WARN] lang is almost entirely NA/None. Если нужен фильтр по языку — придётся детектить в preprocess.")

    if "created_at" in df.columns:
        created = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        print("\ncreated_at range (UTC):", created.min(), "→", created.max())

    if "text" in df.columns:
        lens = df["text"].fillna("").astype(str).str.len()
        print("\ntext length percentiles:")
        print(lens.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

        print("\nExamples (first N rows with text):")
        show = df.loc[df["text"].fillna("").astype(str).str.len() > 0, ["post_id", "subreddit", "text"]].head(args.n)
        for _, row in show.iterrows():
            t = row["text"]
            t_preview = (t[:300] + " ...") if len(t) > 300 else t
            print("-" * 80)
            print("post_id:", row.get("post_id"))
            print("subreddit:", row.get("subreddit"))
            print(t_preview)

    if "subreddit" in df.columns:
        print("\nTop-20 subreddits:")
        print(df["subreddit"].value_counts(dropna=False).head(20))

    print("\nDone.")


if __name__ == "__main__":
    main()
