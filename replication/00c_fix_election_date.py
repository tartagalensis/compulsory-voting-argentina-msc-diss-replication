"""00c — One-shot data repair: election date 2025-10-25 -> 2025-10-26.

The pipeline was built with ELECTION_DATE = 2025-10-25, but the 2025 Argentine
legislative election was held on Sunday 2025-10-26 (the 25th was a Saturday).
The register itself localizes the cutoff: under the stale date, the day-cohort at
days_from_18 == -1 votes at 71.66% -- the treated level (71.24% at 0), not the
control level (53.51% at -2). Those voters turned 18 on election day and were
compelled; the stale running variable placed them on the control side.

ELECTION_DATE enters src/pipeline.py only as a minuend, and date_18/date_70 are
relativedelta offsets from the birth date, so the defect is a uniform -1 day shift
of age_days / days_from_18 / days_from_70 and nothing else. This script applies the
+1 correction in place, which is cheaper than and provably identical to re-running
Stage A (00_build_data.py) -- the equivalence is not assumed, it is CHECKED against
a live Stage-A recomputation of one province before anything is written.

Run once, via:  python replication/run_all.py --fix-date
or standalone:  python replication/00c_fix_election_date.py

Idempotent: refuses to run a second time (see _already_repaired).
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402
from config import PROCESSED_DIR, FINAL_DIR  # noqa: E402
from src.pipeline import load_padron, ELECTION_DATE  # noqa: E402

SHIFT_COLS = ['age_days', 'days_from_18', 'days_from_70']
VALIDATION_PROVINCE = 24          # Tierra del Fuego — the smallest, so this is fast
UNTOUCHED = ['circuit_id', 'province_id', 'voted', 'gender', 'nbi', 'pca_index',
             'pct_sin_secundaria', 'pct_sin_computadora', 'pct_hacinamiento',
             'pct_sin_cloaca', 'pct_sin_piso_tipo1']


def _log(msg):
    """Timestamped, flushed progress line — localizes the failure if a run dies."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _shift(df):
    """Apply the +1 correction to a copy of `df` and return it.

    age is re-derived from the corrected age_days, and compulsory re-derived from
    the corrected running variables, exactly as load_padron does.
    """
    out = df.copy()
    for c in SHIFT_COLS:
        out[c] = out[c] + 1
    out['age'] = out['age_days'] / 365.25
    out['compulsory'] = ((out['days_from_18'] >= 0) & (out['days_from_70'] < 0)).astype(int)
    return out


def _already_repaired(df):
    """True if the file looks repaired already. Last-resort guard only.

    BEFORE the fix, day-cohorts -1 and 0 are BOTH on the treated side (71.66% vs
    71.24% nationally) and so nearly equal. AFTER the fix, -1 holds the old -2 cohort
    (53.51%) and 0 holds the old -1 (71.66%), so they differ by ~18pp.
    Near-equal => not yet repaired. Far apart => already repaired.

    Only consulted for national.parquet, and only when no .oct25.bak survives: it is
    too weak for the province files, where a single day-cohort holds ~100 voters.
    Normal idempotency comes from repair() re-deriving from the backup.
    """
    lo = df.loc[df['days_from_18'] == -1, 'voted']
    hi = df.loc[df['days_from_18'] == 0, 'voted']
    if len(lo) < 200 or len(hi) < 200:
        return False          # too few observations to judge (small provinces)
    return abs(lo.mean() - hi.mean()) > 0.05


def validate_equivalence():
    """Prove the +1 shift equals a Stage-A rebuild, on one province, before mutating.

    Recomputes province VALIDATION_PROVINCE from the raw register at the corrected
    ELECTION_DATE and compares it to the stored parquet with the shift applied.
    Raises SystemExit on any mismatch — if this fails, the shortcut is invalid and
    Stage A must be re-run in full instead.
    """
    _log(f"validating shift-equivalence on province {VALIDATION_PROVINCE} "
         f"(ELECTION_DATE={ELECTION_DATE.date()})")
    if str(ELECTION_DATE.date()) != '2025-10-26':
        raise SystemExit(f"ELECTION_DATE is {ELECTION_DATE.date()}, expected 2025-10-26 "
                         f"— correct src/pipeline.py before running this script.")

    fresh = load_padron(VALIDATION_PROVINCE).reset_index(drop=True)
    stored = pd.read_parquet(
        f"{PROCESSED_DIR}/province_{VALIDATION_PROVINCE:02d}.parquet").reset_index(drop=True)
    shifted = _shift(stored)

    if len(fresh) != len(shifted):
        raise SystemExit(f"row counts differ (raw {len(fresh):,} vs stored "
                         f"{len(shifted):,}) — positional comparison invalid, "
                         f"the shift shortcut cannot be validated. STOP.")

    # NaN-aware comparison: rows with an unparseable birth date are NaN on both
    # sides and must count as a match (NaN != NaN would otherwise flag them all).
    for c in ['age_days', 'days_from_18', 'days_from_70', 'compulsory']:
        a, b = fresh[c], shifted[c]
        n = int((~((a == b) | (a.isna() & b.isna()))).sum())
        _log(f"  {c:<14} mismatches vs Stage-A rebuild: {n:,}  "
             f"(NaN on both sides: {int((a.isna() & b.isna()).sum()):,})")
        if n:
            raise SystemExit(f"{c} does not match a Stage-A rebuild. STOP.")

    if not fresh['age'].isna().equals(shifted['age'].isna()):
        raise SystemExit("age NaN mask does not match a Stage-A rebuild. STOP.")
    amax = float((fresh['age'] - shifted['age']).abs().max())     # skipna
    _log(f"  {'age':<14} max abs diff: {amax:.2e}")
    if amax >= 1e-9:
        raise SystemExit("age does not match a Stage-A rebuild. STOP.")

    _log("EQUIVALENCE PROVEN: +1 shift == Stage-A rebuild at the corrected date")


def repair(path, backup_suffix='.oct25.bak'):
    """Back up, apply the +1 shift, verify, and write `path` in place.

    Idempotent by construction: the backup holds the pristine pre-migration file,
    so once it exists the shift is always re-derived FROM it rather than from the
    current contents. Re-running after a partial failure therefore cannot
    double-shift a file that was already written.
    """
    name = os.path.basename(path)
    bak = path + backup_suffix

    if os.path.exists(bak):
        src = bak                      # re-run: the backup is the pristine original
    else:
        shutil.copy2(path, bak)
        src = path
    df = pd.read_parquet(src)

    out = _shift(df)
    assert out.shape == df.shape, f"{name}: shape changed"

    # A small share of rows carry an unparseable birth date (to_datetime(errors=
    # 'coerce') -> NaT), so every date-derived column is NaN for them: 10,806 rows
    # nationally (0.030%). That is a property of the SOURCE DATA, not of this
    # migration. The contract is that the shift preserves the NaN mask exactly and
    # moves every other row by +1 — NOT that the frame is NaN-free, which it never was.
    for c in SHIFT_COLS:
        assert out[c].isna().equals(df[c].isna()), f"{name}: {c} NaN mask changed"
        d = (out[c] - df[c]).dropna()
        assert len(d) == int(df[c].notna().sum()), f"{name}: {c} lost non-NaN rows"
        assert d.eq(1).all(), f"{name}: {c} did not move by exactly +1"
    assert out['age'].isna().equals(df['age'].isna()), f"{name}: age NaN mask changed"

    for c in UNTOUCHED:
        if c in df.columns:
            assert out[c].equals(df[c]), f"{name}: {c} changed but should not have"

    out.to_parquet(path, index=False)
    _log(f"  repaired {name}: {len(out):,} rows"
         f"{' (re-derived from backup)' if src == bak else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-validation", action="store_true",
                   help="skip the Stage-A equivalence check (which needs the restricted "
                        "raw register); NOT recommended")
    a = p.parse_args()

    # Idempotency pre-flight on the NATIONAL file. repair() is already idempotent
    # whenever a .oct25.bak exists (it re-derives from that pristine copy), so a
    # re-run after a partial failure is safe and must NOT be blocked. The only
    # unguarded case is a completed migration whose backups were later cleaned up:
    # there is then no pristine copy, so fall back to the day-cohort heuristic.
    nat = f"{FINAL_DIR}/national.parquet"
    if not os.path.exists(nat + '.oct25.bak') and \
            _already_repaired(pd.read_parquet(nat, columns=['days_from_18', 'voted'])):
        raise SystemExit("REFUSING: national.parquet looks already repaired and no "
                         ".oct25.bak remains to re-derive from. Not double-shifting.")

    if a.skip_validation:
        _log("WARNING: skipping the Stage-A equivalence check")
    else:
        validate_equivalence()

    for i in range(1, 25):
        repair(f"{PROCESSED_DIR}/province_{i:02d}.parquet")
    repair(nat)

    df = pd.read_parquet(nat, columns=['days_from_18', 'voted'])
    g = df[df['days_from_18'].between(-4, 3)].groupby('days_from_18')['voted'].mean() * 100
    _log("post-repair turnout by day-cohort around the age-18 cutoff:")
    for d, v in g.items():
        _log(f"    days_from_18={int(d):+d}  {v:5.2f}%")
    _log("done — the 18pp jump must now sit between -1 and 0")


if __name__ == '__main__':
    main()
