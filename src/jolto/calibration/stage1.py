"""Stage 1: the rough-SABR marginal calibrator (RoughSABRCalibrator)."""

import numpy as np
import time
import math
import torch
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from jolto.calibration.pricers import (
    bachelier_iv_newton,
    _core_fukasawa_ode,
)
from jolto.core.analytics import (
    generate_sobol_bridge_increments,
    rough_vol_compensator,
)
from jolto import config
from jolto.core.utils import log_progress

from jolto.calibration.pricers import INV_SQRT_2, INV_SQRT_2PI, _hybrid_driver


class RoughSABRCalibrator:
    """Stage 1: fits the rough SABR marginals (ν, α_i, ρ_i) on a fixed grid of Hurst exponents.

    vol_matrix: DataFrame of market normal vols (decimal), index = expiry in years, columns =
    absolute strike offsets from the forward (decimal, 0.005 = 50 bp); NaN marks a missing quote.
    """

    def __init__(self, vol_matrix):
        self.vol_matrix = vol_matrix
        self.expiries = self.vol_matrix.index.values.astype(float)
        self.strike_offsets = self.vol_matrix.columns.values.astype(float)
        self.n_exp = len(self.expiries)

        self.market_vols = self.vol_matrix.values.flatten()
        T_grid, K_grid = np.meshgrid(self.expiries, self.strike_offsets, indexing="ij")
        self.T_flat, self.K_flat = T_grid.flatten(), K_grid.flatten()

        valid = ~np.isnan(self.market_vols)
        self.market_vols = self.market_vols[valid]
        self.T_flat = self.T_flat[valid]
        self.K_flat = self.K_flat[valid]

        self.t_map_idx = np.array([np.argmin(np.abs(self.expiries - t)) for t in self.T_flat])

    def _compute_vega_weights(self):
        market_std = self.market_vols * np.sqrt(self.T_flat)
        market_std = np.maximum(market_std, 1e-12)
        d_market = -self.K_flat / market_std
        raw_weights = np.exp(-0.5 * d_market**2)
        # floor Vega weights so deep-OTM wings keep some pull on the fit
        self.market_weights = np.maximum(raw_weights, 0.25)

    def _seed_global_ode(self, H, base_market_alphas):
        t_seed_ode_start = time.time()

        def seed_obj(p):
            nu_val = p[0]
            rhos_val = p[1:]
            alphas_val = self.exact_atm_alpha(
                self.expiries, base_market_alphas, rhos_val, nu_val, H
            )
            a_flat = alphas_val[self.t_map_idx]
            r_flat = rhos_val[self.t_map_idx]
            v_ode = self.rough_sabr_vol_ode(self.K_flat, self.T_flat, a_flat, r_flat, nu_val, H)
            return (v_ode - self.market_vols) * 10000.0 * np.sqrt(self.market_weights)

        guess_seed = np.concatenate(([0.5], np.zeros(self.n_exp)))
        res_seed = least_squares(
            seed_obj,
            guess_seed,
            bounds=([0.001] + [-0.99] * self.n_exp, [10.0] + [0.99] * self.n_exp),
            method="trf",
            ftol=1e-3,
        )

        smart_nu_guess = res_seed.x[0]
        local_rhos = res_seed.x[1:]
        seed_alphas = self.exact_atm_alpha(
            self.expiries, base_market_alphas, local_rhos, smart_nu_guess, H
        )
        t_seed_ode = time.time() - t_seed_ode_start
        log_progress(
            "Pre-Opt",
            f"ODE seeding done:   {t_seed_ode:.2f}s",
            level=1,
        )
        return smart_nu_guess, local_rhos, seed_alphas, t_seed_ode

    def exact_atm_alpha(self, T, target_atm_vol, rho, nu, H):
        """Return model vol alpha from ATM vols"""
        c_H = 1.0 / (np.sqrt(2.0 * H) * math.gamma(H + 0.5))
        nu_eff = nu * c_H
        drift = (2.0 - 3.0 * rho**2) / 24.0 * nu_eff**2 * (T ** (2.0 * H))
        denom = np.maximum(1.0 + drift, 1e-2)
        return target_atm_vol / denom

    def _generate_mc_cache(self, H, n_paths=config.CALI_PATHS, dt=config.CALI_DT):
        print(
            "      -> [Cache] Starting MC Driver Cache Generation (Bridge+AV Enabled)... ",
            end="",
            flush=True,
        )
        t0 = time.time()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float64

        with torch.no_grad():
            max_T = float(self.expiries.max())
            n_steps = int(math.ceil(max_T / dt))
            print(f" (Steps: {n_steps})...", end="", flush=True)

            # the Sobol+bridge draws do not depend on H: build once per calibrator, reuse per H
            dz_key = (n_paths, n_steps, dt, device)
            if getattr(self, "_dz_vol_key", None) != dz_key:
                dz_all = generate_sobol_bridge_increments(n_paths, n_steps, 1, dt, device, dtype)
                self._dz_vol, self._dz_vol_key = dz_all[..., 0], dz_key
            dz_vol = self._dz_vol
            kappa = config.HYBRID_KAPPA
            fbm, var_vec, kernel = _hybrid_driver(dz_vol, H, n_steps, dt, kappa, device, dtype)

        print(f" Done! ({time.time() - t0:.2f}s)", flush=True)

        return {
            "fbm": fbm,
            "kernel": kernel,
            "dz_vol": dz_vol,
            "var_vec": var_vec,
            "hybrid_kappa": kappa,
            "generation_time_s": float(time.time() - t0),
        }

    def rough_sabr_vol_ode(self, k, T, alpha, rho, nu, H):
        """Normal IV via the Fukasawa & Gatheral (2022) closed-form approximation (G_A, eq. 14)."""
        c_H = 1.0 / (np.sqrt(2.0 * H) * math.gamma(H + 0.5))
        nu_eff = nu * c_H  # nu_eff^2 T^{2H} = nu^2 Var Z~(T), the level correction below

        # Their kappa drives the forward variance xi = alpha^2, so it is twice the log-alpha kernel
        # nu / Gamma(H+1/2): at H = 1/2 the shape factor is then Hagan's normal SABR at vol-of-vol nu.
        kappa_T = 2.0 * nu / math.gamma(H + 0.5) * (T ** (H - 0.5))
        y = kappa_T * k / alpha

        rho_comp = np.sqrt(np.maximum(1.0 - rho**2, 1e-8))

        # G_0: rough limit H -> 0
        def G_0(x):
            term1 = np.log(np.maximum(1.0 + 2.0 * rho * x + x**2, 1e-10))
            term2 = (2.0 * rho / rho_comp) * (
                np.arctan(rho / rho_comp) - np.arctan((x + rho) / rho_comp)
            )
            return term1 + term2

        # G_{1/2}: classical SABR limit H = 1/2
        def G_half(x):
            sq_arg = np.maximum(1.0 + rho * x + (x**2) / 4.0, 1e-10)
            inner = np.sqrt(sq_arg) - rho - x / 2.0
            log_arg = np.maximum(inner / (1.0 - rho), 1e-10)
            return 4.0 * (np.log(log_arg)) ** 2

        h_term1 = 2.0 * H + 1.0
        h_term2 = 2.0 * H + 3.0

        w_0 = 3.0 * (1.0 - 2.0 * H) / h_term2
        w_half = 2.0 * H / h_term2

        g_0_eval = G_0(y / h_term1)
        g_half_eval = G_half(2.0 * y / h_term1)

        G_A = (h_term1**2) * (w_0 * g_0_eval + w_half * g_half_eval)
        G_A = np.maximum(G_A, 1e-12)  # prevent div-by-zero in flat regions

        # shape multiplier |y|/sqrt(G_A); =1 at ATM (y=0)
        shape_multiplier = np.where(np.abs(y) < 1e-8, 1.0, np.abs(y) / np.sqrt(G_A))

        drift = (2.0 - 3.0 * rho**2) / 24.0 * (nu_eff**2) * (T ** (2.0 * H))

        return alpha * shape_multiplier * (1.0 + drift)

    def calibrate(self, method=None, H_grid=None):
        """Fit (ν, α_i, ρ_i) for each H in H_grid (default config.H_GRID) with `method` (default
        config.CALI_MODE); returns the lowest-RMSE H, its parameters and every H's fit under
        "profile_likelihood"."""
        method = config.CALI_MODE if method is None else method
        H_grid = np.asarray(config.H_GRID) if H_grid is None else H_grid
        if method not in ("AMMO_ODE", "PURE_MC", "AMMO_MC"):
            raise ValueError(f"unknown Stage-1 method {method!r}")

        print("\n" + "=" * 60)
        print(f"{f'ROUGH SABR 1D CALIBRATION | MODE: {method.upper()}':^60}")
        print("=" * 60)

        best_rmse = np.inf
        best_H = None
        best_alphas = None
        best_rhos = None
        best_nu = None

        atm_idx = np.argmin(np.abs(self.strike_offsets))
        base_market_alphas = self.vol_matrix.iloc[:, atm_idx].values.copy()

        log_progress("Pre-Opt", "Calculating static Market Vega weights...", level=0)

        self._compute_vega_weights()

        start_time_total = time.time()

        profile_likelihood = []

        for i, H in enumerate(H_grid):
            step_start_time = time.time()
            log_progress(
                "Stage 1", f"Grid {i + 1:2d}/{len(H_grid)} | Testing Hurst (H) = {H:.3f}", level=0
            )

            # initial smart feeding
            smart_nu_guess, local_rhos, seed_alphas, t_seed_ode = self._seed_global_ode(
                H, base_market_alphas
            )
            t_seed_total = t_seed_ode

            if method == "AMMO_ODE":
                # x = [ν, ρ_1..M]; α is pinned to the ATM vol
                guess_global = np.concatenate(([smart_nu_guess], local_rhos))
                lower_bounds = np.concatenate(([0.001], np.full(self.n_exp, -0.999)))
                upper_bounds = np.concatenate(([10.0], np.full(self.n_exp, 0.999)))

                log_progress("AMMO-Opt", "Running fast ODE surrogate optimization...", level=1)

                def global_obj(p):
                    nu = p[0]
                    rhos = p[1:]
                    r_flat = rhos[self.t_map_idx]
                    alphas = self.exact_atm_alpha(self.expiries, base_market_alphas, rhos, nu, H)
                    a_flat = alphas[self.t_map_idx]
                    v_surr = self.rough_sabr_vol_ode(
                        self.K_flat, self.T_flat, a_flat, r_flat, nu, H
                    )
                    return (v_surr - self.market_vols) * 10000.0 * np.sqrt(self.market_weights)

                res = least_squares(
                    global_obj, guess_global, bounds=(lower_bounds, upper_bounds), method="trf"
                )
                current_nu = res.x[0]
                current_rhos = res.x[1:]
                current_alphas = self.exact_atm_alpha(
                    self.expiries, base_market_alphas, current_rhos, current_nu, H
                )

                a_ts = PchipInterpolator(self.expiries, current_alphas, extrapolate=True)
                r_ts = PchipInterpolator(self.expiries, current_rhos, extrapolate=True)
                v_final = self.rough_sabr_vol_ode(
                    self.K_flat, self.T_flat, a_ts(self.T_flat), r_ts(self.T_flat), current_nu, H
                )
                rmse = np.sqrt(np.mean(((v_final - self.market_vols) * 10000.0) ** 2))

            elif method == "PURE_MC":
                log_progress(
                    "MC-Opt", "Starting Optimized Pure Monte Carlo Global Optimization...", level=1
                )

                # x = [ν, α_1..M, ρ_1..M]
                guess_global = np.concatenate(([smart_nu_guess], seed_alphas, local_rhos))
                lower_bounds = np.concatenate(
                    ([0.001], np.full(self.n_exp, 0.0001), np.full(self.n_exp, -0.999))
                )
                upper_bounds = np.concatenate(
                    ([10.0], np.full(self.n_exp, 1.0), np.full(self.n_exp, 0.999))
                )

                eval_counter = [0]

                mc_cache = self._generate_mc_cache(H, n_paths=config.CALI_PATHS)
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # the HF model of AMMO_MC (float32 paths, pathwise Jacobian), so the two
                # methods differ only in the optimiser. One call returns vols and Jacobian;
                # least_squares asks for the Jacobian at the point it just evaluated, so the
                # cache serves it.
                hf_eval = self._make_hf_evaluator(mc_cache, device)
                sqrt_w = np.sqrt(self.market_weights)
                last = {"x": None}

                def _hf(p):
                    if last["x"] is None or not np.array_equal(last["x"], p):
                        last["x"] = np.array(p, copy=True)
                        last["v"], last["J"] = hf_eval(p)
                    return last["v"], last["J"]

                def mc_global_obj(p):
                    eval_counter[0] += 1
                    v_mc, _ = _hf(p)
                    if eval_counter[0] % 10 == 0:
                        print(
                            f"   [MC-Opt] Eval: {eval_counter[0]:>3} | Nu: {p[0]:.4f} | RMSE: {np.sqrt(np.mean((v_mc - self.market_vols) ** 2)) * 10000:.2f}bps",
                            flush=True,
                        )
                    return (v_mc - self.market_vols) * 10000.0 * sqrt_w

                def mc_global_jac(p):
                    _, J = _hf(p)
                    return J * 10000.0 * sqrt_w[:, None]

                res_mc = least_squares(
                    mc_global_obj,
                    guess_global,
                    jac=mc_global_jac,
                    bounds=(lower_bounds, upper_bounds),
                    method="trf",
                    ftol=1e-4,
                    xtol=1e-4,
                )
                print(f"PURE_MC Finished in {res_mc.nfev} HF Evaluations.")

                current_nu = res_mc.x[0]
                current_alphas = res_mc.x[1 : self.n_exp + 1]
                current_rhos = res_mc.x[self.n_exp + 1 :]
                v_final, _ = _hf(res_mc.x)

                # vega-weighted RMSE: PURE_MC and AMMO_MC share this definition (AMMO_ODE does not)
                resid = (v_final - self.market_vols) * 10000.0
                rmse = np.sqrt(np.sum(self.market_weights * resid**2) / np.sum(self.market_weights))

            elif method == "AMMO_MC":
                log_progress(
                    "AMMO-MC-Opt",
                    "Starting Variable-Fidelity Trust-Region Optimization...",
                    level=1,
                )

                initial_params = np.concatenate(([smart_nu_guess], seed_alphas, local_rhos))
                (
                    current_nu,
                    current_alphas,
                    current_rhos,
                    current_hf_rmse,
                    v_hf_accepted,
                    _ammo_timing,
                ) = self._calibrate_ammo_mc_loop(initial_params, H, self.market_weights)

                # report the accepted incumbent, not a fresh re-roll
                v_final = v_hf_accepted
                resid = (v_final - self.market_vols) * 10000.0
                rmse = np.sqrt(np.sum(self.market_weights * resid**2) / np.sum(self.market_weights))

            step_time = time.time() - step_start_time
            # the headline RMSE is vega-weighted; AMMO_MC also logs the unweighted RMSE of the
            # accepted incumbent as a diagnostic
            _rmse_msg = f"Done! RMSE: {rmse:6.2f} bps (vega-wtd)"
            if method == "AMMO_MC":
                _rmse_msg += f" | unwtd: {current_hf_rmse:6.2f} bps"
            log_progress(
                "Stage 1",
                f"{_rmse_msg} | Nu: {current_nu:.4f} | Time: {step_time:5.2f}s\n",
                level=0,
            )

            if method == "AMMO_MC":
                t_hf = _ammo_timing.get("hf_total", 0.0)
                t_sub = _ammo_timing.get("lf_sub_total", 0.0)
                t_beta = _ammo_timing.get("beta_total", 0.0)
                t_cache = _ammo_timing.get("cache", 0.0)
                t_other = max(0.0, step_time - t_seed_total - t_cache - t_hf - t_sub - t_beta)
                log_progress(
                    "Timing",
                    f"Seeding {t_seed_total:5.2f}s | Cache {t_cache:5.2f}s | "
                    f"HF-eval {t_hf:5.2f}s | LF-sub {t_sub:5.2f}s | "
                    f"Beta+FD {t_beta:5.2f}s | Overhead {t_other:5.2f}s",
                    level=0,
                )

            profile_likelihood.append(
                {
                    "H": H,
                    "rmse_bps": rmse,
                    "nu": current_nu,
                    "alphas": current_alphas.copy(),
                    "rhos": current_rhos.copy(),
                    # reuse the v_final already computed for this H's RMSE
                    "v_surface_flat": v_final.copy(),
                }
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_H = H
                best_alphas = current_alphas.copy()
                best_rhos = current_rhos.copy()
                best_nu = current_nu

        total_time = time.time() - start_time_total
        print("\nStatus : SUCCESS")
        print(f"Global Hurst (H): {best_H:.6f}")
        print(f"Global Nu       : {best_nu:.4f}")
        print(f"Best RMSE       : {best_rmse:.4f} bps")
        print(f"1D Calibration Total Time: {total_time:.2f}s")
        print("=" * 60)

        return {
            "alpha_func": PchipInterpolator(self.expiries, best_alphas, extrapolate=True),
            "H": best_H,
            "rmse_bps": best_rmse,
            "rho_func": PchipInterpolator(self.expiries, best_rhos, extrapolate=True),
            "profile_likelihood": profile_likelihood,
            "best_params": {"alphas": best_alphas, "rhos": best_rhos, "nu": best_nu},
        }

    def _make_hf_evaluator(self, mc_cache, device):
        """Stage-1 HF model for one H: CMC vols and their pathwise Jacobian, shared by
        AMMO_MC and PURE_MC so the two methods differ only in the optimiser."""
        # float32 copies of the O(paths x steps) drivers, cast once per H rather than per HF call
        fbm_f32 = mc_cache["fbm"].to(torch.float32).to(device)
        kernel_f32 = mc_cache["kernel"].to(torch.float32).to(device)
        dz_vol_f32 = mc_cache["dz_vol"].to(torch.float32).to(device)
        var_vec_f32 = mc_cache.get("var_vec")
        if var_vec_f32 is not None:
            var_vec_f32 = var_vec_f32.to(torch.float32).to(device)

        def get_hf_vols_and_jac(p_np):
            """HF evaluation: CMC prices + pathwise Jacobian (ν, α, ρ; control-variate β frozen)."""

            def _sync():
                """Force CUDA completion before reading the clock; no-op on CPU."""
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            n_opts = len(self.T_flat)
            print(
                f"      -> [HF Model] Pricing {n_opts} options via CMC (pathwise Jacobian)... ",
                end="",
                flush=True,
            )
            t0 = time.time()

            p_t = torch.tensor(p_np, device=device, dtype=torch.float64)
            n_params = len(p_t)

            nu_base = p_t[0].to(torch.float32)
            # float32 for O(Nxpaths) tensors: halves VRAM bandwidth
            fbm, kernel, dz_vol, var_vec = fbm_f32, kernel_f32, dz_vol_f32, var_vec_f32
            hybrid_kappa = mc_cache.get("hybrid_kappa", 0)

            n_paths = fbm.shape[0]
            dt_t = torch.tensor(config.CALI_DT, device=device, dtype=torch.float32)

            with torch.no_grad():
                var_comp = rough_vol_compensator(nu_base, kernel, var_vec, dt_t, hybrid_kappa)
                vol_exp = torch.exp(nu_base * fbm - var_comp)
                unit_vols = torch.cat(
                    [
                        torch.ones(1, n_paths, 1, device=device, dtype=torch.float32),
                        vol_exp[..., :-1],
                    ],
                    dim=-1,
                )

                max_steps = int(math.ceil(float(self.T_flat.max()) / config.CALI_DT))
                # the (1, 1, n) compensator makes unit_vols (1, paths, n); dz_vol is (paths, n)
                u_vols_trunc = unit_vols.squeeze(0)[:, :max_steps]
                dz_v_trunc = dz_vol[:, :max_steps]

                I_v_base = torch.cumsum(u_vols_trunc * dz_v_trunc, dim=-1)
                I_V2_base = torch.cumsum((u_vols_trunc**2) * dt_t, dim=-1)
                I_cv_v = torch.cumsum(dz_v_trunc, dim=-1)

                T_t = torch.tensor(self.T_flat, device=device, dtype=torch.float64)
                K_t = torch.tensor(self.K_flat, device=device, dtype=torch.float64)
                step_indices = torch.clamp(torch.ceil(T_t / config.CALI_DT).long() - 1, min=0)

                I_v_T_base = I_v_base[:, step_indices]
                I_V2_T_base = I_V2_base[:, step_indices]
                I_cv_v_T = I_cv_v[:, step_indices]

                t_map_t = torch.tensor(self.t_map_idx, device=device, dtype=torch.long)

                _sync()
                t_setup = time.time()

                def _fast_payoff_sliced(
                    alpha, rho, I_v, I_V2, I_cv, k, T, is_call, dI_v_nu, dI_V2_nu
                ):
                    I_v = I_v.to(torch.float64)
                    I_V2 = I_V2.to(torch.float64)
                    I_cv = I_cv.to(torch.float64)
                    dI_v_nu = dI_v_nu.to(torch.float64)
                    dI_V2_nu = dI_V2_nu.to(torch.float64)

                    a_exp = alpha.unsqueeze(0)
                    r_exp = rho.unsqueeze(0)
                    r_comp = torch.sqrt(torch.clamp(1.0 - r_exp**2, min=1e-12))

                    K_exp = k.unsqueeze(0)
                    sign = torch.where(
                        is_call.unsqueeze(0),
                        torch.tensor(1.0, device=device, dtype=torch.float64),
                        torch.tensor(-1.0, device=device, dtype=torch.float64),
                    )

                    sqrt_I_V2 = torch.sqrt(torch.clamp(I_V2, min=1e-12))
                    F_cond = a_exp * r_exp * I_v
                    Std_cond = a_exp * r_comp * sqrt_I_V2

                    d_cond = (F_cond - K_exp) / Std_cond
                    d_sign = torch.where(is_call.unsqueeze(0), d_cond, -d_cond)

                    pdf_cond = INV_SQRT_2PI * torch.exp(-0.5 * d_cond * d_cond)
                    cdf_cond = 0.5 * (1.0 + torch.erf(d_sign * INV_SQRT_2))
                    pay_cond = sign * (F_cond - K_exp) * cdf_cond + Std_cond * pdf_cond

                    F_cv_cond = a_exp * r_exp * I_cv
                    Std_cv_cond = a_exp * r_comp * torch.sqrt(T.unsqueeze(0))

                    d_cv_cond = (F_cv_cond - K_exp) / Std_cv_cond
                    d_cv_sign = torch.where(is_call.unsqueeze(0), d_cv_cond, -d_cv_cond)

                    pdf_cv = INV_SQRT_2PI * torch.exp(-0.5 * d_cv_cond * d_cv_cond)
                    cdf_cv = 0.5 * (1.0 + torch.erf(d_cv_sign * INV_SQRT_2))
                    pay_cv_cond = sign * (F_cv_cond - K_exp) * cdf_cv + Std_cv_cond * pdf_cv

                    # analytical true CV expectation
                    std_cv_true = alpha * torch.sqrt(T)
                    d_cv_true = -k / torch.clamp(std_cv_true, min=1e-12)

                    pdf_true = INV_SQRT_2PI * torch.exp(-0.5 * d_cv_true * d_cv_true)
                    cdf_true_call = 0.5 * (1.0 + torch.erf(d_cv_true * INV_SQRT_2))
                    cdf_true_put = 0.5 * (1.0 + torch.erf(-d_cv_true * INV_SQRT_2))

                    true_cv_call = std_cv_true * (cdf_true_call * d_cv_true + pdf_true)
                    true_cv_put = std_cv_true * (cdf_true_put * (-d_cv_true) + pdf_true)
                    true_cv_otm = torch.where(k >= 0, true_cv_call, true_cv_put)

                    # eq::pmc, with c_cv = Cov/Var estimated on the same paths
                    mean_pay_cond = pay_cond.mean(0, keepdim=True)
                    mean_pay_cv_cond = pay_cv_cond.mean(0, keepdim=True)
                    cov = torch.mean(
                        (pay_cond - mean_pay_cond) * (pay_cv_cond - mean_pay_cv_cond), dim=0
                    )
                    var = torch.var(pay_cv_cond, dim=0)
                    beta_f = cov / torch.clamp(var, min=1e-12)

                    mc_otm_price = torch.mean(
                        pay_cond - beta_f * (pay_cv_cond - true_cv_otm.unsqueeze(0)), dim=0
                    )
                    price = torch.where(k >= 0, mc_otm_price, mc_otm_price - k)

                    # Pathwise sensitivities of the CMC estimator, β frozen. Bachelier:
                    # ∂pay/∂F = sign·Φ(sign·d), ∂pay/∂Std = φ(d); ∂true_cv/∂α = √T·φ(d_true).
                    beta_e = beta_f.unsqueeze(0)
                    delta = sign * cdf_cond
                    delta_cv = sign * cdf_cv
                    sqrt_T = torch.sqrt(T.unsqueeze(0))
                    dpay_da = delta * r_exp * I_v + pdf_cond * r_comp * sqrt_I_V2
                    dcv_da = delta_cv * r_exp * I_cv + pdf_cv * r_comp * sqrt_T
                    grad_alpha = (
                        torch.mean(dpay_da - beta_e * dcv_da, dim=0)
                        + beta_f * torch.sqrt(T) * pdf_true
                    )
                    # ∂r_comp/∂ρ = −ρ/r_comp
                    drc = -r_exp / r_comp
                    dpay_dr = delta * a_exp * I_v + pdf_cond * a_exp * drc * sqrt_I_V2
                    dcv_dr = delta_cv * a_exp * I_cv + pdf_cv * a_exp * drc * sqrt_T
                    grad_rho = torch.mean(dpay_dr - beta_e * dcv_dr, dim=0)
                    # the CV does not depend on ν
                    grad_nu = torch.mean(
                        delta * a_exp * r_exp * dI_v_nu
                        + pdf_cond * a_exp * r_comp * dI_V2_nu / (2.0 * sqrt_I_V2),
                        dim=0,
                    )
                    return price, grad_alpha, grad_rho, grad_nu

                is_call = K_t >= 0
                a_base_flat = p_t[1 : self.n_exp + 1][t_map_t]
                r_base_flat = p_t[self.n_exp + 1 :][t_map_t]

                # ν: pathwise. u = exp(ν·fbm − ½ν²·Var) ⇒ ∂u/∂ν = u·(fbm − ν·Var), with the same
                # left-point shift as u; ν·Var = 2ν·compensator(ν=1) covers both κ branches.
                dvar_comp = (
                    2.0 * nu_base * rough_vol_compensator(1.0, kernel, var_vec, dt_t, hybrid_kappa)
                )
                du_dnu = torch.cat(
                    [
                        torch.zeros(1, n_paths, 1, device=device, dtype=torch.float32),
                        (vol_exp * (fbm - dvar_comp))[..., :-1],
                    ],
                    dim=-1,
                )
                du_trunc = du_dnu.squeeze(0)[:, :max_steps]
                dI_v_nu = torch.cumsum(du_trunc * dz_v_trunc, dim=-1)[:, step_indices]
                dI_V2_nu = torch.cumsum(2.0 * u_vols_trunc * du_trunc * dt_t, dim=-1)[
                    :, step_indices
                ]

                _sync()
                t_nu = time.time()

                mc_prices_t, grad_alpha, grad_rho, grad_nu = _fast_payoff_sliced(
                    a_base_flat,
                    r_base_flat,
                    I_v_T_base,
                    I_V2_T_base,
                    I_cv_v_T,
                    K_t,
                    T_t,
                    is_call,
                    dI_v_nu,
                    dI_V2_nu,
                )

                jac_P_t = torch.zeros((n_opts, n_params), device=device, dtype=torch.float64)
                jac_P_t[:, 0] = grad_nu
                jac_P_t[torch.arange(n_opts), t_map_t + 1] = grad_alpha
                jac_P_t[torch.arange(n_opts), t_map_t + 1 + self.n_exp] = grad_rho

            _sync()
            t_cmc = time.time()

            dt_setup = t_setup - t0
            dt_nu = t_nu - t_setup
            dt_cmc = t_cmc - t_nu
            dt_total_mc = t_cmc - t0
            print(
                f"Done ({dt_total_mc:.2f}s) "
                f"[Setup {dt_setup:.2f}s | "
                f"ν-PW {dt_nu:.2f}s | "
                f"CMC+α/ρ-PW {dt_cmc:.2f}s]"
                f" | Inverting IV & Chain Rule... ",
                end="",
                flush=True,
            )
            t1 = time.time()

            a_ts_guess_t = p_t[1 : self.n_exp + 1][self.t_map_idx]
            v_hf_t = bachelier_iv_newton(mc_prices_t, K_t, T_t, initial_guess_vol=a_ts_guess_t)

            std = torch.clamp(v_hf_t * torch.sqrt(T_t), min=1e-12)
            d = -K_t / std
            pdf_vega = INV_SQRT_2PI * torch.exp(-0.5 * d * d)
            vega = torch.clamp(torch.sqrt(T_t) * pdf_vega, min=1e-12)

            J_hf_t = jac_P_t / vega.unsqueeze(1)

            v_hf_np = v_hf_t.cpu().numpy()
            J_hf = J_hf_t.cpu().numpy()

            print(f"Done ({time.time() - t1:.2f}s).", flush=True)
            return v_hf_np, J_hf

        return get_hf_vols_and_jac

    def _calibrate_ammo_mc_loop(self, initial_params, H, market_weights):
        """AMMO trust-region loop (Alexandrov 2001): HF=rough SABR MC, LF=Fukasawa ODE."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_exp = self.n_exp

        delta = config.TR_DELTA_0
        delta_max = config.TR_DELTA_MAX
        r1, r2 = config.TR_R1, config.TR_R2
        c1, c2 = config.TR_C1, config.TR_C2

        lower_bounds = np.concatenate(([0.001], np.full(n_exp, 0.0001), np.full(n_exp, -0.999)))
        upper_bounds = np.concatenate(([10.0], np.full(n_exp, 1.0), np.full(n_exp, 0.999)))

        mc_cache = self._generate_mc_cache(H, n_paths=config.CALI_PATHS)
        dtype = torch.float64
        K_t = torch.tensor(self.K_flat, device=device, dtype=dtype)
        T_t = torch.tensor(self.T_flat, device=device, dtype=dtype)
        market_vols_t = torch.tensor(self.market_vols, device=device, dtype=dtype)
        market_weights_t = torch.tensor(market_weights, device=device, dtype=dtype)
        t_map_t = torch.tensor(self.t_map_idx, device=device, dtype=torch.long)
        H_t = torch.tensor(H, device=device, dtype=dtype)

        get_hf_vols_and_jac = self._make_hf_evaluator(mc_cache, device)

        def get_lf_vols(p):
            nu, alphas, rhos = p[0], p[1 : n_exp + 1], p[n_exp + 1 :]

            a_flat = alphas[self.t_map_idx]
            r_flat = rhos[self.t_map_idx]

            return self.rough_sabr_vol_ode(self.K_flat, self.T_flat, a_flat, r_flat, nu, H)

        current_x = np.clip(initial_params.copy(), lower_bounds, upper_bounds)
        t_hf_start = time.time()
        v_hf_current, J_hf_current = get_hf_vols_and_jac(current_x)
        t_hf_total = time.time() - t_hf_start
        t_lf_sub_total = 0.0
        t_beta_total = 0.0
        current_hf_rmse = np.sqrt(np.mean(((v_hf_current - self.market_vols) * 10000.0) ** 2))
        merit_current = 0.5 * np.sum(
            ((v_hf_current - self.market_vols) * 10000.0 * np.sqrt(market_weights)) ** 2
        )

        w_sum = float(np.sum(market_weights))
        max_iters = config.AMMO_MAX_ITER
        consecutive_rejects = 0

        for k in range(max_iters):
            print(f"\n   >>> AMMO Outer Iteration {k + 1}/{max_iters} <<<", flush=True)
            t_beta_start = time.time()

            v_lf_at_xk = get_lf_vols(current_x)

            epsilon = 1e-6
            J_lf = np.zeros((len(self.K_flat), len(current_x)))
            for i in range(len(current_x)):
                p_bump = current_x.copy()
                p_bump[i] += epsilon
                J_lf[:, i] = (get_lf_vols(p_bump) - v_lf_at_xk) / epsilon

            beta_k = v_hf_current / v_lf_at_xk

            v_hf_col = v_hf_current[:, np.newaxis]
            v_lf_col = v_lf_at_xk[:, np.newaxis]
            # eq::grad_beta, quotient rule: dβ/dp = (J_hf * v_lf - v_hf * J_lf) / v_lf²
            grad_beta_k = (J_hf_current * v_lf_col - v_hf_col * J_lf) / (v_lf_col**2)
            t_beta_iter = time.time() - t_beta_start
            t_beta_total += t_beta_iter
            print(
                f"      [Beta+FD] Correction computed: {t_beta_iter:.2f}s  "
                f"({len(current_x)} params x FD)",
                flush=True,
            )

            sub_eval_counter = [0]

            beta_k_t = torch.tensor(beta_k, device=device, dtype=dtype)
            grad_beta_k_t = torch.tensor(grad_beta_k, device=device, dtype=dtype)
            current_x_t = torch.tensor(current_x, device=device, dtype=dtype)

            def _diff_subproblem_obj(x_trial_t):
                nu_t = x_trial_t[0]
                alphas_t = x_trial_t[1 : n_exp + 1]
                rhos_t = x_trial_t[n_exp + 1 :]

                a_flat_t = alphas_t[t_map_t]
                r_flat_t = rhos_t[t_map_t]

                v_lf_trial_t = _core_fukasawa_ode(K_t, T_t, a_flat_t, r_flat_t, nu_t, H_t)

                # AMMO first-order Taylor expansion of beta
                step_t = x_trial_t - current_x_t
                beta_trial_t = beta_k_t + torch.mv(grad_beta_k_t, step_t)

                v_surr_t = beta_trial_t * v_lf_trial_t
                return (v_surr_t - market_vols_t) * 10000.0 * torch.sqrt(market_weights_t)

            def subproblem_obj(x_trial):
                x_t = torch.tensor(x_trial, device=device, dtype=dtype)
                with torch.no_grad():
                    res = _diff_subproblem_obj(x_t).cpu().numpy()

                if sub_eval_counter[0] % 10 == 0:
                    nu_trial = x_trial[0]
                    alphas_trial = x_trial[1 : n_exp + 1]
                    rhos_trial = x_trial[n_exp + 1 :]
                    print(
                        f"      [AMMO-Sub] Eval: {sub_eval_counter[0]:>3} | Nu: {nu_trial:.4f} | Mean Alpha: {np.mean(alphas_trial) * 10000:.2f}bps | Mean Rho: {np.mean(rhos_trial):.4f}",
                        flush=True,
                    )

                sub_eval_counter[0] += 1
                return res

            def subproblem_jac(x_trial):
                x_t = torch.tensor(x_trial, device=device, dtype=dtype, requires_grad=True)
                # vmap'd reverse mode: one batched backward instead of one per residual row,
                # bit-identical to the loop. Forward mode is faster per call but its ~1e-11
                # difference flips trust-region decisions.
                jac_t = torch.autograd.functional.jacobian(
                    _diff_subproblem_obj, x_t, vectorize=True
                )
                jac_np = jac_t.detach().cpu().numpy()
                # guard against isolated div-by-zero artifacts
                if not np.isfinite(jac_np).all():
                    jac_np = np.nan_to_num(jac_np, nan=0.0, posinf=0.0, neginf=0.0)
                return jac_np

            # eq:subproblem: ‖p‖_∞ ≤ Δ box around the anchor, clipped to the global bounds
            tr_lower = np.maximum(lower_bounds, current_x - delta)
            tr_upper = np.minimum(upper_bounds, current_x + delta)

            t_sub_start = time.time()
            res_lf = least_squares(
                subproblem_obj,
                current_x,
                jac=subproblem_jac,
                bounds=(tr_lower, tr_upper),
                method="trf",
                ftol=1e-5,
                xtol=1e-5,
                max_nfev=100,
            )
            t_sub_iter = time.time() - t_sub_start
            t_lf_sub_total += t_sub_iter
            print(
                f"      [LF-Sub]  Solved in {t_sub_iter:.2f}s  "
                f"({res_lf.nfev} evals, {sub_eval_counter[0]} obj calls)",
                flush=True,
            )

            x_trial = res_lf.x

            t_hf_iter_start = time.time()
            v_hf_trial, J_hf_trial = get_hf_vols_and_jac(x_trial)
            t_hf_total += time.time() - t_hf_iter_start
            trial_hf_rmse = np.sqrt(np.mean(((v_hf_trial - self.market_vols) * 10000.0) ** 2))

            # ‖p_k‖_∞, taken before the accept rebinds current_x
            step_norm = np.linalg.norm(x_trial - current_x, ord=np.inf)

            # call surrogate objective once; result reused for ψ_k and termination
            surr_resid_w = subproblem_obj(x_trial)

            # ½‖·‖²_W merit with the same Vega weights in numerator and denominator of ψ_k
            merit_trial_hf = 0.5 * np.sum(
                ((v_hf_trial - self.market_vols) * 10000.0 * np.sqrt(market_weights)) ** 2
            )
            merit_pred = 0.5 * np.sum(surr_resid_w**2)

            # accept and stop on the weighted merit that ψ_k uses, expressed as the
            # vega-weighted RMSE in bps so the 1e-3 / 1e-2 tolerances keep their bps scale.
            # A rejection implies ψ_k ≤ 0 < r₁, so the radius always contracts on reject.
            w_rmse_current = np.sqrt(2.0 * merit_current / w_sum)
            w_rmse_trial = np.sqrt(2.0 * merit_trial_hf / w_sum)
            actual_reduction = w_rmse_current - w_rmse_trial
            predicted_reduction = w_rmse_current - np.sqrt(2.0 * merit_pred / w_sum)
            merit_actual = merit_current - merit_trial_hf
            merit_predicted = merit_current - merit_pred
            merit_pred_safe = (
                merit_predicted
                if abs(merit_predicted) > 1e-12
                else np.copysign(1e-12, merit_predicted)
            )
            r = merit_actual / merit_pred_safe  # ψ_k, eq:psi

            status_str = "ACCEPTED" if actual_reduction > 0 else "REJECTED"
            print(
                f"      [TR]      r={r:+.3f} | {status_str} | "
                f"ΔRMSE_w={actual_reduction:+.4f} bps | "
                f"RMSE_w {w_rmse_current:.3f} -> {w_rmse_trial:.3f} bps | "
                f"unwtd {current_hf_rmse:.3f} -> {trial_hf_rmse:.3f} | "
                f"δ={delta:.4f}",
                flush=True,
            )

            if actual_reduction > 0:
                current_x = x_trial
                v_hf_current = v_hf_trial
                J_hf_current = J_hf_trial
                current_hf_rmse = trial_hf_rmse
                merit_current = merit_trial_hf
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1

            if r < r1:
                delta = max(1e-5, c1 * step_norm)
            elif r > r2:
                delta = min(c2 * delta, delta_max)

            # termination: surrogate predicts no meaningful improvement
            if predicted_reduction < 1e-3:
                break
            # termination: HF improvement negligible
            if 0 < actual_reduction < 1e-2:
                break
            # termination: consecutive rejects or trust-region collapse
            if consecutive_rejects >= 2:
                break
            if delta <= 1e-3:
                break

        _cache_time = mc_cache.get("generation_time_s", 0.0)
        _timing = {
            "cache": _cache_time,
            "hf_total": t_hf_total,
            "lf_sub_total": t_lf_sub_total,
            "beta_total": t_beta_total,
        }
        return (
            current_x[0],
            current_x[1 : n_exp + 1],
            current_x[n_exp + 1 :],
            current_hf_rmse,
            # accepted-incumbent HF vols, so calibrate() reports the accepted state, not a re-roll
            v_hf_current,
            _timing,
        )
