"""Build results/comparison.md and results/comparison.png from results/summary.json.

Usage: python -m eval.compare && python -m eval.report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"
TYPES = ["gender", "name", "college"]


def main() -> None:
    s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    lines = ["| Metric | v1 | v2 |", "|---|---|---|"]

    def row(label, fn):
        lines.append(f"| {label} | {fn(s['v1'])} | {fn(s['v2'])} |")

    row("Valid pairs", lambda x: f"{x['bias']['pairs_valid']}/{x['bias']['pairs_total']}")
    row("Flip rate (all)", lambda x: f"{x['bias']['overall']['flip_rate']:.1%}")
    for t in TYPES:
        row(f"Flip rate ({t})", lambda x, t=t: f"{x['bias']['by_type'].get(t, {}).get('flip_rate', float('nan')):.1%}")
    row("Mean abs score gap", lambda x: f"{x['bias']['overall']['mean_abs_score_gap']:.2f}")
    row("McNemar p", lambda x: f"{x['bias']['overall']['mcnemar']['p_value']:.3f}")
    row("Wilcoxon p", lambda x: f"{x['bias']['overall']['wilcoxon']['p_value']:.3f}")
    row("Accuracy (labelled set)", lambda x: f"{x['accuracy']['accuracy']:.1%}")
    (RES / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    w = 0.38
    xs = range(len(TYPES))
    for i, v in enumerate(("v1", "v2")):
        vals = [100 * s[v]["bias"]["by_type"].get(t, {}).get("flip_rate", 0) for t in TYPES]
        ax.bar([x + (i - 0.5) * w for x in xs], vals, w, label=v)
    ax.set_xticks(list(xs), TYPES)
    ax.set_ylabel("Flip rate (%)")
    ax.set_title("Decision flips per pair type (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "comparison.png", dpi=150)
    print("wrote results/comparison.md and results/comparison.png")


if __name__ == "__main__":
    main()
