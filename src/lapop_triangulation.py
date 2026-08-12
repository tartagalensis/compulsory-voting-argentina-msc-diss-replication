"""
lapop_triangulation.py
----------------------
Independent survey corroboration of the female turnout advantage discussed in the
gender-mechanism chapter (Chapter 7). Computes self-reported turnout by sex from
the 2023 AmericasBarometer (Argentina): variable ``vb2`` (reported vote in the
2019 presidential election) by ``q1tc_r`` (sex), using the country survey weight
``wt`` (uniform in the country file, so weighted == unweighted).

Source: LAPOP Lab, AmericasBarometer Argentina 2023.

Run:
    python src/lapop_triangulation.py

The .dta lives under data/raw/ (not tracked by git); the path is resolved relative
to the repository root so the script is machine-independent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "lapop_2023"
    / "ARG_2023_LAPOP_AmericasBarometer_v1.0_w.dta"
)

SEX_LABELS = {1: "Male", 2: "Female"}


def self_reported_turnout_by_sex(path=DATA_PATH):
    """Weighted self-reported turnout (share) by sex, with the female-male gap.

    Returns a dict with keys 'Male', 'Female', 'n', and 'gap_F_minus_M'.
    """
    df = pd.read_stata(path, convert_categoricals=False)
    d = df[["vb2", "q1tc_r", "wt"]].copy()
    # vb2: 1 = voted, 2 = did not vote; q1tc_r: 1 = male, 2 = female
    d = d[d["vb2"].isin([1, 2]) & d["q1tc_r"].isin([1, 2])]
    d["voted"] = (d["vb2"] == 1).astype(int)

    out = {}
    for sex, g in d.groupby("q1tc_r"):
        out[SEX_LABELS[int(sex)]] = float(np.average(g["voted"], weights=g["wt"]))
    out["n"] = int(len(d))
    out["gap_F_minus_M"] = out["Female"] - out["Male"]
    return out


if __name__ == "__main__":
    r = self_reported_turnout_by_sex()
    print(f"n = {r['n']}")
    print(f"Male   turnout (self-report): {r['Male'] * 100:.1f}%")
    print(f"Female turnout (self-report): {r['Female'] * 100:.1f}%")
    print(f"Female - Male gap:            {r['gap_F_minus_M'] * 100:+.1f} pp")

    # Validate against the figures cited in the dissertation (Chapter 7).
    assert r["n"] == 1513, r["n"]
    assert round(r["Male"] * 100, 1) == 81.7, r["Male"]
    assert round(r["Female"] * 100, 1) == 86.3, r["Female"]
    assert round(r["gap_F_minus_M"] * 100, 1) == 4.6, r["gap_F_minus_M"]
    print("\nOK: reproduces the figures cited in Chapter 7.")
