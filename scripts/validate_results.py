"""Validate experimental CSV integrity, without training or third-party dependencies.

Usage: python scripts/validate_results.py experiment_results.csv
"""

import csv
import math
import sys
from pathlib import Path

REQUIRED = {
    "model", "support_size", "train_user_fraction", "n_train_users",
    "train_pool_size", "ndcg_mean", "ndcg_std", "n_episodes",
    "train_time_sec", "inference_time_sec", "simclr_train_time_sec",
}
MODELS = {"knn", "linear_regression", "matching_network", "reptile"}
SUPPORT_SIZES = {5, 10, 20}


def validate(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"Missing CSV columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        rows = list(reader)

    if not rows:
        raise ValueError("Results CSV contains no observations.")
    seen = set()
    for number, row in enumerate(rows, start=2):
        model = row["model"]
        support = int(row["support_size"])
        fraction = float(row["train_user_fraction"])
        if model not in MODELS or support not in SUPPORT_SIZES:
            raise ValueError(f"Row {number}: unexpected model/support-size combination")
        key = (model, support, fraction)
        if key in seen:
            raise ValueError(f"Row {number}: duplicate experimental condition {key}")
        seen.add(key)
        if not 0 < fraction <= 1:
            raise ValueError(f"Row {number}: training fraction out of range")
        mean = float(row["ndcg_mean"])
        std = float(row["ndcg_std"])
        if not 0 <= mean <= 1 or not 0 <= std <= 1:
            raise ValueError(f"Row {number}: NDCG metric out of range")
        for field in ("n_train_users", "train_pool_size", "n_episodes"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0 or not value.is_integer():
                raise ValueError(f"Row {number}: invalid positive count {field}")
        for field in ("train_time_sec", "inference_time_sec", "simclr_train_time_sec"):
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Row {number}: negative/nonfinite time {field}")
        if int(row["n_train_users"]) > int(row["train_pool_size"]):
            raise ValueError(f"Row {number}: more training users than eligible pool")
    return len(rows), len({float(r["train_user_fraction"]) for r in rows})


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiment_results.csv")
    try:
        count, fractions = validate(target)
    except (ValueError, OSError, KeyError) as exc:
        print(f"INVALID results: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Valid results: {count} conditions over {fractions} train-user fraction(s).")
