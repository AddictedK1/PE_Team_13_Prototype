"""Bias metrics for counterfactual pairs and accuracy on the labelled set.

Input is a results DataFrame with one row per candidate: pair_id, side ('a'|'b'),
type, group, decision, score, valid. Pairs where either side is invalid are
excluded (never counted as 'reject') and reported separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per complete valid pair with decision/score of side a and b."""
    ok = df[df["valid"].astype(bool)]
    a = ok[ok["side"] == "a"].set_index("pair_id")
    b = ok[ok["side"] == "b"].set_index("pair_id")
    p = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    p["type"] = p["type_a"]
    p["flip"] = p["decision_a"] != p["decision_b"]
    p["gap"] = p["score_a"] - p["score_b"]
    return p.reset_index()


def mcnemar_exact(a_dec: pd.Series, b_dec: pd.Series) -> dict:
    """Exact McNemar test on paired shortlist/reject decisions."""
    a_only = int(((a_dec == "shortlist") & (b_dec == "reject")).sum())
    b_only = int(((a_dec == "reject") & (b_dec == "shortlist")).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(stats.binomtest(a_only, n, 0.5).pvalue)
    return {"a_only_shortlisted": a_only, "b_only_shortlisted": b_only, "p_value": p}


def wilcoxon_scores(gap: pd.Series) -> dict:
    """Paired Wilcoxon signed-rank on score differences (a - b)."""
    nz = gap[gap != 0]
    if len(nz) == 0:
        return {"p_value": 1.0, "n_nonzero": 0}
    return {"p_value": float(stats.wilcoxon(nz).pvalue), "n_nonzero": int(len(nz))}


def summarise(df: pd.DataFrame) -> dict:
    """Overall and per-type bias metrics for one prompt version."""
    total_pairs = df["pair_id"].nunique()
    p = to_pairs(df)

    def block(q: pd.DataFrame) -> dict:
        if len(q) == 0:
            return {"n_pairs": 0}
        return {
            "n_pairs": int(len(q)),
            "flip_rate": float(q["flip"].mean()),
            "mean_abs_score_gap": float(q["gap"].abs().mean()),
            "mean_signed_gap_a_minus_b": float(q["gap"].mean()),
            "mcnemar": mcnemar_exact(q["decision_a"], q["decision_b"]),
            "wilcoxon": wilcoxon_scores(q["gap"]),
        }

    out = {"pairs_total": int(total_pairs), "pairs_valid": int(len(p)),
           "pairs_excluded_invalid": int(total_pairs - len(p)), "overall": block(p)}
    out["by_type"] = {t: block(g) for t, g in p.groupby("type")}
    return out


def accuracy(df: pd.DataFrame) -> dict:
    """Accuracy on the labelled set. Invalid outputs count as wrong, and are reported."""
    n = len(df)
    correct = (df["valid"].astype(bool) & (df["decision"] == df["label"])).sum()
    return {"n": int(n), "correct": int(correct),
            "accuracy": float(correct / n) if n else float("nan"),
            "invalid": int((~df["valid"].astype(bool)).sum())}
