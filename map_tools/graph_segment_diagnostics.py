import json
import os
from collections import Counter


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "map_outputs")
MAPPING_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_stage2_mapping.json")


def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    fallback = os.path.basename(path)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(path)


def main():
    with open(_resolve(MAPPING_PATH), "r", encoding="utf-8") as fh:
        mapping = json.load(fh)

    pair_counts = Counter()
    pair_lengths = {}

    for item in mapping.get("edge_mapping", []):
        u = item["train_u"]
        v = item["train_v"]
        key = (u, v) if u <= v else (v, u)
        pair_counts[key] += 1
        pair_lengths.setdefault(key, []).append(float(item.get("length", 0.0)))

    count_hist = Counter(pair_counts.values())
    ratio_hist = Counter()

    for lengths in pair_lengths.values():
        if len(lengths) < 2:
            continue
        mn = min(lengths)
        mx = max(lengths)
        if mn <= 1e-9:
            continue
        ratio = mx / mn
        if ratio < 1.2:
            ratio_hist["<1.2"] += 1
        elif ratio < 1.5:
            ratio_hist["1.2-1.5"] += 1
        elif ratio < 2.0:
            ratio_hist["1.5-2.0"] += 1
        else:
            ratio_hist[">=2.0"] += 1

    print("Segment count per cluster-pair")
    for k in sorted(count_hist):
        print(f"  {k}: {count_hist[k]}")

    print("\nLength ratio histogram (only cluster-pairs with >=2 segments)")
    for label in ["<1.2", "1.2-1.5", "1.5-2.0", ">=2.0"]:
        print(f"  {label}: {ratio_hist.get(label, 0)}")


if __name__ == "__main__":
    main()
