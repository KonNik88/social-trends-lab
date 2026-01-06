import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/raw/reddit_hf_mix_300k.parquet")
    ap.add_argument("--out_path", default="data/raw/reddit_hf_mix_30k.parquet")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    in_path = root / args.in_path
    out_path = root / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_parquet(in_path)
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved sample: {out_path} | rows={len(df):,}")


if __name__ == "__main__":
    main()
