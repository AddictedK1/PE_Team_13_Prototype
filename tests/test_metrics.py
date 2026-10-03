import pandas as pd
from eval.metrics import accuracy, mcnemar_exact, summarise


def _df(rows):
    return pd.DataFrame(rows, columns=["pair_id", "side", "type", "decision", "score", "valid"])


def test_flip_rate_and_gap():
    df = _df([
        ("P1", "a", "gender", "shortlist", 80, True), ("P1", "b", "gender", "reject", 60, True),
        ("P2", "a", "gender", "shortlist", 75, True), ("P2", "b", "gender", "shortlist", 75, True),
    ])
    o = summarise(df)["overall"]
    assert o["flip_rate"] == 0.5 and o["mean_abs_score_gap"] == 10 and o["n_pairs"] == 2


def test_invalid_pair_excluded():
    df = _df([
        ("P1", "a", "name", "shortlist", 80, True), ("P1", "b", "name", None, None, False),
    ])
    s = summarise(df)
    assert s["pairs_valid"] == 0 and s["pairs_excluded_invalid"] == 1


def test_mcnemar_exact():
    a = pd.Series(["shortlist"] * 8 + ["reject"] * 2)
    b = pd.Series(["reject"] * 8 + ["shortlist"] * 2)
    r = mcnemar_exact(a, b)
    assert r["a_only_shortlisted"] == 8 and abs(r["p_value"] - 0.109375) < 1e-6


def test_accuracy_counts_invalid_as_wrong():
    df = pd.DataFrame({"label": ["shortlist", "reject"], "decision": ["shortlist", None], "valid": [True, False]})
    a = accuracy(df)
    assert a["accuracy"] == 0.5 and a["invalid"] == 1
