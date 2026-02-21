import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_map(scores_csv, keywords_tsv):
    scores = pd.read_csv(scores_csv)
    keywords = pd.read_csv(keywords_tsv, sep="\t", header=None, names=["id", "keyword"])
    merged = scores.merge(keywords, left_on="query_id", right_on="id", suffixes=("", "_query"))
    merged = merged.merge(keywords, left_on="target_id", right_on="id", suffixes=("", "_target"))

    aps = []
    for query_id, group in merged.groupby("query_id"):
        query_word = group["keyword_query"].iloc[0]
        group = group.sort_values("distance", ascending=True)
        group["relevant"] = group["keyword_target"] == query_word
        rel = group["relevant"].to_numpy().astype(int)

        if rel.sum() == 0:
            continue

        precisions = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        ap = (precisions * rel).sum() / rel.sum()
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def plot_pr_curve(scores_csv, keywords_tsv, out_path="results/pr_curve.png"):
    scores = pd.read_csv(scores_csv)
    keywords = pd.read_csv(keywords_tsv, sep="\t", header=None, names=["id", "keyword"])
    merged = scores.merge(keywords, left_on="query_id", right_on="id", suffixes=("", "_query"))
    merged = merged.merge(keywords, left_on="target_id", right_on="id", suffixes=("", "_target"))

    for query_id, group in merged.groupby("query_id"):
        query_word = group["keyword_query"].iloc[0]
        group = group.sort_values("distance", ascending=True)
        group["relevant"] = group["keyword_target"] == query_word
        rel = group["relevant"].to_numpy().astype(int)
        if rel.sum() == 0:
            continue
        precisions = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        recalls = np.cumsum(rel) / rel.sum()
        plt.figure()
        plt.plot(recalls, precisions, marker="o")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve for query '{query_word}'")
        plt.grid(True)
        Path(out_path).parent.mkdir(exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Saved PR curve for '{query_word}' to {out_path}")
        break
