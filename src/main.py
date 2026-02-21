from pathlib import Path
from preprocess import load_polygons, preprocess_word
from features import extract_features
from dtw import dtw_distance
from evaluate import compute_map, plot_pr_curve
import numpy as np
import csv

def process_image(image_path, svg_path):
    polygons = load_polygons(str(svg_path))
    if len(polygons) < 2:
        return []

    ids = list(polygons.keys())
    words = [preprocess_word(str(image_path), polygons[i]) for i in ids]
    features = [extract_features(w) for w in words]

    query_idx = 0
    query_id = ids[query_idx]
    query_feat = features[query_idx]
    distances = []

    for j, f in enumerate(features):
        if j == query_idx:
            continue
        d = dtw_distance(query_feat, f)

        print(f"[DEBUG] {query_id} vs {ids[j]} -> {d:.4f}")

        distances.append((query_id, ids[j], float(d)))

    distances.sort(key=lambda x: x[2])
    return distances



def main():
    data_dir = Path("kws")
    images_dir = data_dir / "images"
    locations_dir = data_dir / "locations"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_results = []
    images = sorted(images_dir.glob("*.jpg"))

    for img in images:
        svg = locations_dir / (img.stem + ".svg")
        if not svg.exists():
            print(f"[WARN] Missing SVG for {img.name}, skipping.")
            continue

        print(f"[INFO] Processing {img.name}")
        distances = process_image(img, svg)
        all_results.extend(distances)

    out_path = results_dir / "scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "target_id", "distance"])
        writer.writerows(all_results)

    print(f"[DONE] Saved results to {out_path}")

    keywords_tsv = data_dir / "keywords.tsv"
    if keywords_tsv.exists():
        map_score = compute_map(out_path, keywords_tsv)
        print(f"[FINAL] Mean Average Precision (mAP): {map_score:.3f}")
        plot_pr_curve(out_path, keywords_tsv, out_path="results/pr_curve.png")
    else:
        print("[WARN] keywords.tsv not found — skipping mAP and PR plot.")


if __name__ == "__main__":
    main()
