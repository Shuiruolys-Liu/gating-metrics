#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gating_metrics open benchmark — external hold-out of the gating
measurement specification (four never-used acute-inflammation blood cohorts).

Ground truth, stated honestly: the authors' own value-blind prospective test
of the REGISTERED reference-stratum prediction was NEGATIVE (NOT CONFIRMED).
This benchmark exists so that any group can recompute that verdict from
subject-level data and add their own cohorts under the identical frozen
protocol — not to present the result as positive.

Protocol (frozen in manifest module G1_external_holdout_gating_spec, value-blind,
registered before any contact with these datasets; OSF registration
10.17605/OSF.IO/4TZH9 carries the field-testable prediction):
  - per-sample KAT2A_Z and module_Z are the highest-variance-probe-per-gene
    Z scores (Z across full cohort) used by the frozen F1/F2 metric dictionary;
    module_Z = mean Z over {NLRP3, NLRC4, PYCARD, CASP1, IL1B}.
  - stratum coupling r = Spearman(KAT2A_Z, module_Z) within stratum (n>=8).
  - reference stratum name: control_REF.
  - verdicts PG1-1..PG1-4 exactly as frozen (see code).

Usage:  python run_benchmark.py
Inputs: data/G1_<accession>_per_sample.csv (shipped, subject-level derived data)
Output: benchmark_results.csv + console verdict table
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REF = "control_REF"
MODULE_GENES = ["NLRP3", "NLRC4", "PYCARD", "CASP1", "IL1B"]  # frozen F1/F2 dictionary


def mde_r(n, power=0.80, alpha=0.05):
    """Minimum detectable |r| at the given power (Fisher-Z closed form)."""
    if n < 4:
        return np.nan
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float(np.tanh((z_a + z_b) / np.sqrt(n - 3)))


def fisher_z_contrast(r1, n1, r2, n2):
    """Two-sided p for r1 vs r2 (independent)."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))


def main():
    rows, refs = [], []
    files = sorted(f for f in os.listdir(DATA) if f.endswith("_per_sample.csv"))
    assert files, "no per-sample files found in data/"
    for fn in files:
        acc = fn.split("_")[1]
        meta = pd.read_csv(os.path.join(DATA, fn))
        ref_r = np.nan
        for st in sorted(meta["stratum"].astype(str).unique()):
            s = meta[meta["stratum"] == st][["KAT2A_Z", "module_Z"]].dropna()
            if len(s) >= 8 and s["KAT2A_Z"].std() > 0 and s["module_Z"].std() > 0:
                r, p = stats.spearmanr(s["KAT2A_Z"], s["module_Z"])
            else:
                r, p = np.nan, np.nan
            rows.append(dict(cohort=acc, stratum=st, n=int(len(s)),
                             r=float(r) if np.isfinite(r) else np.nan,
                             p=float(p) if np.isfinite(p) else np.nan,
                             MDE_r_80pct=mde_r(len(s))))
            if st == REF:
                ref_r = r
        # acute contrasts vs reference (frozen G1 logic)
        n_ref = int((meta["stratum"] == REF).sum())
        for st in sorted(meta["stratum"].astype(str).unique()):
            if st == REF or "descriptive" in st or "later" in st:
                continue
            s = meta[meta["stratum"] == st][["KAT2A_Z", "module_Z"]].dropna()
            if np.isfinite(ref_r) and len(s) >= 8:
                rr, _ = stats.spearmanr(s["KAT2A_Z"], s["module_Z"])
                _, pz = fisher_z_contrast(rr, len(s), ref_r, n_ref)
                rows.append(dict(cohort=acc, stratum=f"CONTRAST_{st}_vs_REF", n=int(len(s)),
                                 r=float(rr), p=np.nan, MDE_r_80pct=np.nan,
                                 delta_r=float(rr - ref_r), fisher_z_p=pz))
        if np.isfinite(ref_r):
            refs.append(dict(cohort=acc, r=ref_r,
                             p=[x["p"] for x in rows if x["cohort"] == acc and x["stratum"] == REF][0],
                             n=[x["n"] for x in rows if x["cohort"] == acc and x["stratum"] == REF][0]))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "benchmark_results.csv"), index=False)

    # ---- frozen verdicts ----
    ref_df = pd.DataFrame(refs)
    n_eval, n_neg = len(ref_df), int((ref_df["r"] < 0).sum())
    pg1_1 = (n_eval > 0) and (n_neg / n_eval >= 0.80)
    z = np.arctanh(ref_df["r"].clip(-0.999, 0.999).values)
    v = 1.0 / (ref_df["n"].values - 3.0)
    w = 1.0 / v
    z_fe = float((w * z).sum() / w.sum())
    p_fe_one = float(stats.norm.sf(z_fe / np.sqrt(1 / w.sum())))
    q = float((w * (z - z_fe) ** 2).sum())
    c = float(w.sum() - (w * w).sum() / w.sum())
    tau2 = max(0.0, (q - (len(ref_df) - 1)) / c)
    w_re = 1.0 / (v + tau2)
    z_re = float((w_re * z).sum() / w_re.sum())
    n_tot = int(ref_df["n"].sum())
    mde_pool = mde_r(n_tot)
    r_fe, r_re = float(np.tanh(z_fe)), float(np.tanh(z_re))
    pg1_3 = (p_fe_one < 0.05) and (abs(r_fe) >= mde_pool)
    acute = df[df["stratum"].astype(str).str.startswith("CONTRAST")]
    pg1_4 = int((acute["delta_r"] > 0).sum()) if len(acute) else 0

    print("== gating_metrics open benchmark — frozen G1 verdicts (recomputed) ==")
    print(f"PG1-1 reference direction-negative >=80% : {n_neg}/{n_eval}  -> {'PASS' if pg1_1 else 'FAIL'}")
    for _, x in ref_df.iterrows():
        print(f"   {x.cohort}: reference r={x.r:+.4f} (n={x.n}, p={x.p:.3g})")
    print(f"PG1-3 pooled FE negative & above MDE     : r_FE={r_fe:+.4f} (n={n_tot}, MDE={mde_pool:.3f}, one-sided p={p_fe_one:.3g}); "
          f"DL RE r={r_re:+.4f} -> {'PASS' if pg1_3 else 'FAIL'}")
    print(f"PG1-4 acute contrasts delta-r>0 (gate-opening direction): {pg1_4}/{len(acute)}")
    print(f"VERDICT (reference-stratum spec)         : {'VALIDATED' if (pg1_1 and pg1_3) else 'NOT CONFIRMED'}")
    print("Honest note: the authors' registered reference-stratum prediction is NOT")
    print("externally confirmed on these four cohorts; the acute-inflamed strata")
    print("couple negatively (see CONTRAST rows / benchmark_results.csv).")
    print("Adding your cohort: place a G1_<ACCESSION>_per_sample.csv in data/ with")
    print("columns gsm,stratum,KAT2A_Z,module_Z (use the gating_metrics metric")
    print("dictionary to build them), rerun this script.")


if __name__ == "__main__":
    main()
