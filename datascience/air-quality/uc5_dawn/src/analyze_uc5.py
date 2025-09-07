# analyze_uc5.py — robust to split_veh / split_haze columns

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV  = Path(__file__).resolve().parents[1] / "results" / "uc5_features.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT    = OUT_DIR / "haze_vs_density.png"
SUMMARY = OUT_DIR / "SUMMARY.md"

def make_split_column(df: pd.DataFrame) -> pd.DataFrame:
    # If a unified 'split' exists, keep it; otherwise build one.
    if "split" in df.columns:
        return df

    veh_has  = "split_veh"  in df.columns
    haze_has = "split_haze" in df.columns

    if veh_has and haze_has:
        # If the two splits match, use either; otherwise prefer veh and fill from haze
        equal_mask = (df["split_veh"].astype(str) == df["split_haze"].astype(str))
        if equal_mask.all():
            df["split"] = df["split_veh"]
        else:
            df["split"] = df["split_veh"].fillna(df["split_haze"])
            df["split"] = df["split"].fillna("mixed")
    elif veh_has:
        df["split"] = df["split_veh"].fillna("all")
    elif haze_has:
        df["split"] = df["split_haze"].fillna("all")
    else:
        df["split"] = "all"

    return df

def main():
    print(f"Loaded: {IN_CSV}")
    df = pd.read_csv(IN_CSV)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # Basic checks
    needed = {"vehicles", "haze"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    # Build unified split
    df = make_split_column(df)

    print("\nColumns:", df.columns.tolist())
    print("\nHead:\n", df.head())

    # Per-split summary (count, mean±std)
    print("\nPer-split summary:")
    g = df.groupby("split")[["vehicles", "haze"]].agg(["count", "mean", "std"]).round(3)
    print(g)

    # Save a simple summary to markdown
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write("# UC5: Vehicles vs Haze — Summary\n\n")
        f.write(f"- Rows: **{len(df)}**\n\n")
        try:
            corr = df["vehicles"].corr(df["haze"])
        except Exception:
            corr = float("nan")
        f.write(f"- Pearson corr (vehicles, haze): **{corr:.4f}**\n\n")
        f.write("## Per-split stats\n\n")
        f.write(g.to_markdown())

    # Scatter plot
    plt.figure(figsize=(7,5))
    plt.scatter(df["vehicles"], df["haze"], s=10, alpha=0.45)
    plt.title("Haze vs Vehicle Density")
    plt.xlabel("Vehicles per image")
    plt.ylabel("Haze score (DCP)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT, dpi=150)
    print(f"\nSaved plot to: {PLOT}")
    print(f"Saved summary to: {SUMMARY}")

if __name__ == "__main__":
    main()
