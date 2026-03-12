import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def generate_balanced_dataset(
    n_total: int,
    seed: int,
    output_path: Path,
) -> pd.DataFrame:
    """
    Generate a roughly class-balanced dataset:
      - 0: Safe
      - 1: Warning
      - 2: Danger
    """
    if n_total <= 0:
        raise ValueError("n_total must be positive")

    np.random.seed(seed)

    # Split total samples as evenly as possible across 3 classes
    n_each, remainder = divmod(n_total, 3)
    counts = np.array([n_each, n_each, n_each], dtype=int)
    # Distribute any remainder to the first few classes
    counts[:remainder] += 1

    n_safe, n_warning, n_danger = counts.tolist()

    # --- SAFE SAMPLES ---
    # All sensors comfortably in normal range
    safe = np.column_stack(
        [
            np.random.uniform(0, 50, n_safe),       # PM25 low
            np.random.uniform(0, 150, n_safe),      # VOC low
            np.random.uniform(20, 32, n_safe),      # HeatIdx normal
            np.random.uniform(55, 90, n_safe),      # HR normal
            np.random.uniform(96, 100, n_safe),     # SpO2 normal
        ]
    )
    safe_labels = np.zeros(n_safe, dtype=int)

    # --- WARNING SAMPLES ---
    # At least one sensor elevated but not dangerous
    warning = np.column_stack(
        [
            np.random.uniform(55, 149, n_warning),   # PM25 elevated
            np.random.uniform(150, 600, n_warning),  # VOC elevated
            np.random.uniform(32, 40, n_warning),    # HeatIdx warm
            np.random.uniform(90, 130, n_warning),   # HR elevated
            np.random.uniform(93, 96, n_warning),    # SpO2 slightly low
        ]
    )
    warning_labels = np.ones(n_warning, dtype=int)

    # --- DANGER SAMPLES ---
    # At least one sensor in dangerous range
    danger = np.column_stack(
        [
            np.random.uniform(150, 500, n_danger),   # PM25 dangerous
            np.random.uniform(600, 1000, n_danger),  # VOC dangerous
            np.random.uniform(40, 50, n_danger),     # HeatIdx dangerous
            np.random.uniform(130, 180, n_danger),   # HR dangerous
            np.random.uniform(85, 93, n_danger),     # SpO2 dangerous
        ]
    )
    danger_labels = np.full(n_danger, 2, dtype=int)

    # Combine all three classes
    X = np.vstack([safe, warning, danger])
    y = np.concatenate([safe_labels, warning_labels, danger_labels])

    # Shuffle so classes aren't in order
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    df = pd.DataFrame(
        {
            "PM25": X[:, 0],
            "VOC": X[:, 1],
            "HeatIdx": X[:, 2],
            "HR": X[:, 3],
            "SpO2": X[:, 4],
            "label": y,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Class stats using bincount (fast, single pass)
    counts = np.bincount(y, minlength=3)
    total = len(y)

    labels = ["Safe", "Warning", "Danger"]
    print()
    for i, name in enumerate(labels):
        c = counts[i]
        pct = c / total * 100 if total > 0 else 0.0
        print(f"{name:7}: {c:6d} ({pct:5.1f}%)")

    print(f"Total:  {total}")
    print(f"dataset.csv saved to {output_path}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a class-balanced synthetic sensor dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-total",
        type=int,
        default=10_000,
        help="Total number of samples to generate (will be split across 3 classes).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_balanced_dataset(
        n_total=args.n_total,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()