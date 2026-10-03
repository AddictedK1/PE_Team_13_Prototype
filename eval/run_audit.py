"""Run all counterfactual pairs and the labelled set through a prompt version.

Usage: python -m eval.run_audit --version v1 [--provider mock] [--delay 0.5]
Writes results/<version>_results.csv (pairs) and results/<version>_labelled.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from app.llm_client import get_llm_client
from app.screen import screen

ROOT = Path(__file__).resolve().parent.parent
RETRIES = 2  # README: invalid output retried up to 2 times, then flagged


def _row(res) -> dict:
    return {"decision": res.decision, "score": res.score, "reason": res.reason,
            "valid": res.valid, "retried": res.retried, "error": res.error}


def run(version: str, provider: str | None, delay: float, limit: int | None = None) -> None:
    client = get_llm_client(provider=provider)
    pairs = json.loads((ROOT / "data" / "pairs.json").read_text(encoding="utf-8"))[:limit]
    labelled = json.loads((ROOT / "data" / "labelled_set.json").read_text(encoding="utf-8"))[:limit]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    rows = []
    for p in pairs:
        for side in ("a", "b"):
            c = p[side]
            res = screen(c["text"], prompt_version=version, client=client, max_retries=RETRIES)
            rows.append({"pair_id": p["pair_id"], "side": side, "type": p["type"],
                         "group": c["group"], "label": p["label"], **_row(res)})
            time.sleep(delay)
    pd.DataFrame(rows).to_csv(out / f"{version}_results.csv", index=False)

    rows = []
    for r in labelled:
        res = screen(r["text"], prompt_version=version, client=client, max_retries=RETRIES)
        rows.append({"id": r["id"], "difficulty": r["difficulty"], "label": r["label"], **_row(res)})
        time.sleep(delay)
    pd.DataFrame(rows).to_csv(out / f"{version}_labelled.csv", index=False)
    print(f"{version}: wrote results/{version}_results.csv and results/{version}_labelled.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v1", "v2"])
    ap.add_argument("--provider", default=None, help="gemini | openai | mock")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="first N pairs/resumes only (smoke test)")
    a = ap.parse_args()
    run(a.version, a.provider, a.delay, a.limit)
