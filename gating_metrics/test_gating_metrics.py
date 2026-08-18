#!/usr/bin/env python
"""gating_metrics self-tests (frozen F3 deliverable): run with
    python test_gating_metrics.py
Asserts (i) the method-of-moments partition equals the direct algebraic
identity on a synthetic two-level dataset (seed 42), (ii) the closed-form
power calculator reproduces the standard Fisher-Z n formula, (iii) the
platform Fisher-Z contrast recovers a known sign reversal, and (iv) the
compatibility-report classifier matches the frozen thresholds.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from partition import partition, bootstrap_share_between
from power import n_for_coupling, power_at_n, mde_at_n
from platform_contrast import fisher_z_contrast, compatibility_report, atlas_verdict
from coupling import spearman_np, partial_spearman


def make_two_level(seed=42, n_samples=60, cells_per=40, r_between=-0.6,
                   r_within=0.0):
    rng = np.random.default_rng(seed)
    mus = rng.multivariate_normal([0, 0], [[1, r_between], [r_between, 1]],
                                  size=n_samples)
    rows = []
    for s, (mx, my) in enumerate(mus):
        xy = rng.multivariate_normal([mx, my], [[1, r_within], [r_within, 1]],
                                     size=cells_per)
        for x, y in xy:
            rows.append({"sample": f"S{s}", "x": x, "y": y})
    return pd.DataFrame(rows)


def main():
    # (i) partition identity + between-individual share recovery
    df = make_two_level()
    p = partition(df[["sample", "x", "y"]], by=("sample",))
    assert abs(p["SSP_tot"] - p["SSP_between_individual"] - p["SSP_within"]) \
        < 1e-6 * max(1.0, abs(p["SSP_tot"]))
    lo, hi = bootstrap_share_between(df[["sample", "x", "y"]], by=("sample",),
                                     n_boot=200)
    assert lo < p["share_between_individual"] < hi, (lo, p, hi)
    print(f"partition OK: share_between={p['share_between_individual']:.3f} "
          f"bootstrap95=[{lo:.3f},{hi:.3f}]")

    # (ii) power calculators
    n_req = n_for_coupling(0.3)
    assert power_at_n(0.3, n_req) >= 0.8 - 1e-6
    assert abs(mde_at_n(n_req) - 0.3) < 0.02
    print(f"power OK: n(r=0.3, 80%)={n_req}, MDE at that n={mde_at_n(n_req):.3f}")

    # (iii) Fisher-Z platform contrast with known sign reversal
    c = fisher_z_contrast(-0.5, 40, +0.4, 200)
    assert c["sign_reversal"] and c["p"] < 0.05
    print(f"fisher-z OK: p={c['p']:.2e}, sign_reversal={c['sign_reversal']}")

    # (iv) classification thresholds + atlas verdict
    rows = [
        dict(dataset="A", platform="array", compartment="whole_blood",
             state_class="acute_blood", stratum="ref", r=-0.4, p=0.01, n=50),
        dict(dataset="B", platform="array", compartment="whole_blood",
             state_class="acute_blood", stratum="ref", r=-0.3, p=0.02, n=80),
        dict(dataset="C", platform="scRNA", compartment="PBMC",
             state_class="acute_blood", stratum="ref", r=+0.3, p=0.03, n=100),
        dict(dataset="D", platform="array", compartment="whole_blood",
             state_class="acute_blood", stratum="ref", r=-0.1, p=0.4, n=30),
    ]
    rep = compatibility_report(rows)
    v = atlas_verdict(rep, ["acute_blood"])
    assert v["k"] == 2 and v["n"] == 4 and not v["supported"]
    print("classification OK:", dict(zip(rep.dataset, rep.classification)))
    print("ALL gating_metrics SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
