"""Compare v1 vs v2 from results/*.csv and write results/summary.json.

Usage: python -m eval.compare
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eval.metrics import accuracy, summarise

RES = Path(__file__).resolve().parent.parent / "results"


def load_summary(version: str) -> dict:
    pairs = pd.read_csv(RES / f"{version}_results.csv")
    lab = pd.read_csv(RES / f"{version}_labelled.csv")
    return {"bias": summarise(pairs), "accuracy": accuracy(lab)}


if __name__ == "__main__":
    summary = {v: load_summary(v) for v in ("v1", "v2")}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for v, s in summary.items():
        o = s["bias"]["overall"]
        print(f"{v}: pairs {s['bias']['pairs_valid']}/{s['bias']['pairs_total']} valid | "
              f"flip rate {o.get('flip_rate', float('nan')):.1%} | "
              f"mean |gap| {o.get('mean_abs_score_gap', float('nan')):.2f} | "
              f"McNemar p={o['mcnemar']['p_value']:.3f} | Wilcoxon p={o['wilcoxon']['p_value']:.3f} | "
              f"accuracy {s['accuracy']['accuracy']:.1%} ({s['accuracy']['correct']}/{s['accuracy']['n']})")
    print("Note: 42 pairs (14 per type) is low power; a non-significant p is not proof of no bias.")
