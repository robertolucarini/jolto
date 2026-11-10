"""
Engine-A statistical-bias & finite-sample convergence diagnostic.

Engine A = the 1-D conditional-MC Rough SABR calibration pricer
(jolto.calibration.pricers). An independent high-N **PRNG oracle**
(oracle_rough_sabr_pricer, unseeded → batches i.i.d.) is the ground truth; the production
estimators are validated against it with a Welch t-test (Satterthwaite df). Two estimators:
  • is_brute:     in-sample QMC+CV (mc_rough_sabr_pricer);
  • split_sample: pilot-boundary / pilot-CV wrapper (oos_mc_rough_sabr_pricer).
Neither estimator is "out-of-sample" in the sense of tests/oos.py.

Two modes, param source is the only switch (config.CONVERGENCE):
  (a) production: per-expiry α/ρ/ν from the calibrated Stage-1 surface at H=VAL_H_TARGET;
  (b) sweep:      flat-α grid over a config (ρ,ν) lattice (oracle is cheap → fuller grid).

Adds a path-doubling SE-vs-N ladder at the ATM point. Every SE is across scrambles (sample sd of
the m_seeds scramble means over √m_seeds), never sd/√N on one run, whose N^{-1/2} rate holds by
construction. Raw = the same scrambles with β:=0, so the variance ratio is paired. The CV/raw
cost ratio at N_prod is timed on a raw-only path that skips the CV twin.

float32 note: the Engine-A pricers are float32-locked. The float32 reduction floor is
negligible and the driver-independent float32 integrand offset
cancels in bias = estimator_f32 − oracle_f32 (both float32).

Usage:  python tests/test_statistical_bias.py [--full]
Outputs: results/mc_convergence_engineA.csv, results/mc_convergence_engineA_cost.json
"""

import json
import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from jolto import config, val_config
from jolto.calibration.pricers import (
    _cmc_vol_integrals,
    _cond_bachelier_otm,
    mc_rough_sabr_pricer,
    oos_mc_rough_sabr_pricer,
    oracle_rough_sabr_pricer,
)
from jolto.core.analytics import build_hybrid_rough_driver, generate_sobol_bridge_increments
from _common import _cli_profile, _slope, _welch

DEVICE = "cpu"
DT = config.CALI_DT  # Stage-1 calibration grid (Engine A)
H = val_config.VAL_H_TARGET  # production pricing-H
SEED = val_config.PRICE_SEED  # base seed; per-scramble = SEED+s
N_PROD = config.CALI_PATHS  # production path count (top of the ladder)
PILOT_FRAC = val_config.PRICE_PATHS_PILOT / val_config.PRICE_PATHS_MAIN  # split-sample pilot ratio
CFG = config.CONVERGENCE
STAGE1_CSV = os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV)

ESTIMATORS = ("is_brute", "split_sample")


def _t_ci(per_batch, level):
    """t-based CI of the grand mean from per-batch price vectors (n_batches, n_opts)."""
    nb = per_batch.shape[0]
    m = per_batch.mean(0)
    se = per_batch.std(0, ddof=1) / math.sqrt(nb)
    tcrit = stats.t.ppf(0.5 + level / 2.0, df=nb - 1)
    return m, se, m - tcrit * se, m + tcrit * se, nb - 1


def _run_oracle(K, T, alpha, rho, nu, total, batch):
    """Batched PRNG ground truth → per-option (mean, se, ci_lo, ci_hi, df). bps."""
    nb = total // batch
    per_batch = []
    for _ in range(nb):
        with torch.no_grad():
            p = oracle_rough_sabr_pricer(
                K, T, alpha, rho, nu, H, n_paths=batch, dt=DT, device=DEVICE
            )
        per_batch.append(p.cpu().numpy())
    return _t_ci(np.asarray(per_batch) * 1e4, CFG["ci_level"])


def _estimator_means(K, T, alpha, rho, nu, N, m_seeds, kind):
    """m_seeds independent scrambles → (m_seeds, n_opts) prices in bps."""
    out = []
    for s in range(m_seeds):
        with torch.no_grad():
            if kind in ("is_brute", "raw"):
                # raw: β frozen to 0 on the same scramble, so it pairs with is_brute
                beta0 = _zero_beta(K) if kind == "raw" else None
                p = mc_rough_sabr_pricer(
                    K,
                    T,
                    alpha,
                    rho,
                    nu,
                    H,
                    n_paths=N,
                    dt=DT,
                    device=DEVICE,
                    seed=SEED + s,
                    precomputed_beta=beta0,
                )
            else:
                p = oos_mc_rough_sabr_pricer(
                    K,
                    T,
                    alpha,
                    rho,
                    nu,
                    H,
                    n_paths_main=N,
                    n_paths_pilot=max(int(N * PILOT_FRAC), 256),
                    dt=DT,
                    device=DEVICE,
                    seed=SEED + s,
                )
        out.append(p.cpu().numpy())
    return np.asarray(out) * 1e4


def _zero_beta(K):
    return torch.zeros(len(np.atleast_1d(K)), device=DEVICE, dtype=torch.float32)


def _raw_price(K, T, alpha, rho, nu, N, seed):
    """mc_rough_sabr_pricer stopped before the control-variate twin: the raw estimator's own cost.
    Mirrors the pricer's uncached path; _cost_ratio checks it against the β=0 price bit for bit."""
    dtype = torch.float32
    K_t, T_t, a_t, r_t = (
        torch.as_tensor(x, device=DEVICE, dtype=dtype) for x in (K, T, alpha, rho)
    )
    n_steps = int(math.ceil(float(T_t.max()) / DT))
    dz_vol = generate_sobol_bridge_increments(N, n_steps, 1, DT, DEVICE, dtype, seed=seed)[..., 0]
    kappa = config.HYBRID_KAPPA
    near_gen = torch.Generator(device=DEVICE).manual_seed(seed + 7919) if kappa > 0 else None
    fbm, var_vec, kernel = build_hybrid_rough_driver(
        dz_vol, H, n_steps, DT, kappa, DEVICE, dtype, near_generator=near_gen, antithetic=True
    )
    _, _, I_v_T, I_V2_T = _cmc_vol_integrals(
        dz_vol, fbm, var_vec, kernel, nu, T_t, N, n_steps, DT, kappa, DEVICE, dtype
    )
    a_e, r_e, K_e = a_t.unsqueeze(0), r_t.unsqueeze(0), K_t.unsqueeze(0)
    is_call = (K_t >= 0).unsqueeze(0)
    sign = torch.where(is_call, torch.tensor(1.0, dtype=dtype), torch.tensor(-1.0, dtype=dtype))
    std = (
        a_e
        * torch.sqrt(torch.clamp(1.0 - r_e**2, min=1e-12))
        * torch.sqrt(torch.clamp(I_V2_T, min=1e-12))
    )
    otm = _cond_bachelier_otm(a_e * r_e * I_v_T, std, K_e, is_call, sign).mean(0)
    return torch.where(K_t >= 0, otm, otm - K_t).to(torch.float64)


def _cost_ratio(K, T, alpha, rho, nu, N, label, n_warmup=2, n_repeat=9):
    """Median wall-clock of the CV pricer over the raw-only path at one cell, with environment."""
    args = (K, T, alpha, rho, nu, H)
    with torch.no_grad():

        def cv():
            return mc_rough_sabr_pricer(*args, n_paths=N, dt=DT, device=DEVICE, seed=SEED)

        def raw():
            return _raw_price(K, T, alpha, rho, nu, N, SEED)

        ref = mc_rough_sabr_pricer(
            *args, n_paths=N, dt=DT, device=DEVICE, seed=SEED, precomputed_beta=_zero_beta(K)
        )
        if not torch.equal(raw(), ref):
            raise AssertionError("raw-only timing path differs from the β=0 pricer")
        for _ in range(n_warmup):
            raw(), cv()
        ts = {"raw": [], "cv": []}
        # interleaved, so drift in machine load hits both arms alike
        for _ in range(n_repeat):
            for k, fn in (("raw", raw), ("cv", cv)):
                t0 = time.perf_counter()
                fn()
                ts[k].append(time.perf_counter() - t0)
    med = {k: float(np.median(v)) for k, v in ts.items()}
    return {
        "instrument": label,
        "N": int(N),
        "n_warmup": n_warmup,
        "n_repeat": n_repeat,
        "times_s": ts,
        "median_s": med,
        "min_s": {k: float(min(v)) for k, v in ts.items()},
        "max_s": {k: float(max(v)) for k, v in ts.items()},
        "cost_ratio_cv_over_raw": med["cv"] / med["raw"],
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "cpu_count": os.cpu_count(),
            "device": DEVICE,
            "timed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def _eval_cells(param_source, K, T, alpha, rho, nu, N, m_seeds, labels, orc):
    """Assemble CSV rows for every (option, estimator) at one N against a precomputed oracle."""
    orc_m, orc_se, orc_lo, orc_hi, orc_df = orc
    raw = _estimator_means(K, T, alpha, rho, nu, N, m_seeds, "raw")
    se_raw = raw.std(0, ddof=1) / math.sqrt(m_seeds)
    rows = []
    for kind in ESTIMATORS:
        est = _estimator_means(K, T, alpha, rho, nu, N, m_seeds, kind)  # (m_seeds, n_opts)
        est_m = est.mean(0)
        est_se = est.std(0, ddof=1) / math.sqrt(m_seeds)
        for i, lab in enumerate(labels):
            bias = est_m[i] - orc_m[i]
            t, df, p = _welch(bias, est_se[i], m_seeds - 1, orc_se[i], orc_df)
            rows.append(
                {
                    "param_source": param_source,
                    "instrument": lab,
                    "estimator": kind,
                    "N": N,
                    "price": est_m[i],
                    "se_scr": est_se[i],
                    "se_scr_raw": se_raw[i],
                    "vr_scr": (se_raw[i] / est_se[i]) ** 2,
                    "oracle_price": orc_m[i],
                    "bias_bps": bias,
                    "welch_t": t,
                    "welch_df": df,
                    "p_value": p,
                    "in_oracle_CI": bool(orc_lo[i] <= est_m[i] <= orc_hi[i]),
                }
            )
    return rows


def _build_production_grid():
    """Per-expiry α/ρ and a single ν from the calibrated Stage-1 surface at H=VAL_H_TARGET."""
    df = pd.read_csv(STAGE1_CSV)
    dfh = df[np.abs(df["H"] - H) < 1e-6]
    if dfh.empty:
        raise ValueError(f"No Stage-1 params at H={H} in {STAGE1_CSV}")
    a_of = {float(r.Expiry): float(r.Alpha) for r in dfh.itertuples()}
    r_of = {float(r.Expiry): float(r.Rho) for r in dfh.itertuples()}
    if dfh["Nu"].nunique() != 1:
        raise ValueError(f"Stage-1 Nu is not a single value at H={H}: {dfh['Nu'].unique()}")
    nu = float(dfh["Nu"].mean())
    exp = [e for e in CFG["expiries_A"] if e in a_of]
    K_off = np.array(CFG["strike_offsets_A"])
    Tg, Kg = np.meshgrid(np.array(exp), K_off, indexing="ij")
    T_flat, K_flat = Tg.flatten(), Kg.flatten()
    alpha = np.array([a_of[t] for t in T_flat])
    rho = np.array([r_of[t] for t in T_flat])
    labels = [f"{t:.0f}Y_{k * 1e4:+.0f}bp" for t, k in zip(T_flat, K_flat)]
    return K_flat, T_flat, alpha, rho, np.array([nu]), labels


def main():
    prof_key, prof = _cli_profile(CFG)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 84)
    print(f"{'ENGINE A — STATISTICAL BIAS & MC CONVERGENCE (1-D conditional-MC)':^84}")
    print("=" * 84)
    print(
        f"  profile={prof_key}  H={H}  dt={DT}  seed={SEED}  N_prod={N_PROD}  "
        f"oracle_N={prof['oracle_n_A']:,}  M_seeds={prof['m_seeds']}"
    )

    # the oracle draws from the global generator; seeding it makes reruns repeat (batches stay i.i.d.)
    torch.manual_seed(SEED)
    rows = []

    # Mode (a): production point, full-grid bias table at N_prod
    K, T, alpha, rho, nu, labels = _build_production_grid()
    print(
        f"\n[mode a] production grid: {len(labels)} options "
        f"({len(set(T))} expiries × {len(CFG['strike_offsets_A'])} strikes)"
    )
    orc = _run_oracle(K, T, alpha, rho, nu, prof["oracle_n_A"], prof["oracle_batch_A"])
    print(f"  oracle done — max SE {orc[1].max():.4f} bps (df={orc[4]})")
    rows += _eval_cells("production", K, T, alpha, rho, nu, N_PROD, prof["m_seeds"], labels, orc)

    # Mode (a): SE-vs-N ladder at the ATM point
    atm_T = CFG["atm_expiry_A"]
    j = labels.index([lab for lab in labels if lab.startswith(f"{atm_T:.0f}Y_+0bp")][0])
    Ka, Ta, aa, ra = (np.array([K[j]]), np.array([T[j]]), np.array([alpha[j]]), np.array([rho[j]]))
    orc_atm = tuple(np.array([x[j]]) if hasattr(x, "__len__") else x for x in orc)
    print(f"\n[mode a] SE-vs-N ladder @ ATM {atm_T:.0f}Y: N={prof['n_ladder_A']}")
    ladder_se = []
    for N in prof["n_ladder_A"]:
        r = _eval_cells("ladder_atm", Ka, Ta, aa, ra, nu, N, prof["m_seeds"], [labels[j]], orc_atm)
        rows += r
        cv = r[0]  # is_brute
        ladder_se.append((cv["se_scr_raw"], cv["se_scr"]))
        print(
            f"    N={N:7d}  se_scr_raw={cv['se_scr_raw']:.4f}  se_scr_cv={cv['se_scr']:.4f}  "
            f"VR={cv['vr_scr']:.2f}"
        )
    Ns = np.array(prof["n_ladder_A"])
    sl_raw = _slope(Ns, np.array([x[0] for x in ladder_se]))
    sl = _slope(Ns, np.array([x[1] for x in ladder_se]))
    print(f"  across-scramble SE log-log slope: raw {sl_raw:+.3f}, cv {sl:+.3f}")

    cost = _cost_ratio(Ka, Ta, aa, ra, nu, N_PROD, labels[j])
    with open(os.path.join(config.RESULTS_DIR, "mc_convergence_engineA_cost.json"), "w") as f:
        json.dump(cost, f, indent=2)
    print(
        f"  cost CV/raw @ N={N_PROD}: {cost['cost_ratio_cv_over_raw']:.3f}  "
        f"(median {cost['median_s']['cv']:.3f}s / {cost['median_s']['raw']:.3f}s)"
    )

    # Mode (b): representative (ρ,ν) sweep at the ATM expiries (flat α)
    print("\n[mode b] (ρ,ν) sweep — flat α, ATM strikes")
    K_off0 = np.array([0.0])
    exp = np.array(CFG["expiries_A"])
    sweep_labels = [f"{t:.0f}Y_ATM" for t in exp]
    in_ci = []
    for rho_v in CFG["sweep_rho_A"]:
        for nu_v in CFG["sweep_nu_A"]:
            Ks = np.repeat(K_off0, len(exp))
            Ts = exp.copy()
            al = np.full(len(exp), CFG["sweep_alpha"])
            rh = np.full(len(exp), rho_v)
            tag = f"rho{rho_v:+.2f}_nu{nu_v:.2f}"
            o = _run_oracle(
                Ks, Ts, al, rh, np.array([nu_v]), prof["oracle_n_A"], prof["oracle_batch_A"]
            )
            r = _eval_cells(
                f"sweep[{tag}]",
                Ks,
                Ts,
                al,
                rh,
                np.array([nu_v]),
                N_PROD,
                prof["m_seeds"],
                sweep_labels,
                o,
            )
            rows += r
            in_ci += [x["in_oracle_CI"] for x in r]
            mx = max(abs(x["bias_bps"]) for x in r)
            print(
                f"    {tag}: max|bias|={mx:.3f} bps  in-CI {sum(x['in_oracle_CI'] for x in r)}/{len(r)}"
            )
    print(
        f"  [sweep summary] {sum(in_ci)}/{len(in_ci)} cells within the oracle {CFG['ci_level']:.0%} CI"
    )

    df = pd.DataFrame(rows)
    out = os.path.join(config.RESULTS_DIR, "mc_convergence_engineA.csv")
    df.to_csv(out, index=False)

    prod = df[df.param_source == "production"]
    print(f"\n{'=' * 84}\n  PRODUCTION-POINT UNBIASEDNESS (mode a, N={N_PROD})")
    for kind in ESTIMATORS:
        sub = prod[prod.estimator == kind]
        print(
            f"    {kind:>13}: mean|bias|={sub.bias_bps.abs().mean():.3f} bps  "
            f"min p={sub.p_value.min():.3f}  in-CI {int(sub.in_oracle_CI.sum())}/{len(sub)}  "
            f"→ {'INDISTINGUISHABLE from 0' if sub.p_value.min() > 0.05 else 'SIGNIFICANT cell present'}"
        )
    print(f"  across-scramble SE slope: raw {sl_raw:+.3f}, cv {sl:+.3f}.  wrote {out}\n")


if __name__ == "__main__":
    main()
