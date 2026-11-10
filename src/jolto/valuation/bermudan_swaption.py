import torch
from jolto import config
from jolto.core.utils import log_progress
from jolto.core.analytics import generate_sobol_bridge_increments
import time
import warnings

_INV_SQRT_2 = 0.7071067811865476
_INV_SQRT_2PI = 0.3989422804014327


def _bachelier_total(F, K, total_std):
    """Bachelier call on the total std (not vol·√T): the exact mean of (N(F, total_std²) − K)⁺."""
    std = torch.clamp(total_std, min=1e-12)
    d = (F - K) / std
    cdf = 0.5 * (1.0 + torch.erf(d * _INV_SQRT_2))
    pdf = _INV_SQRT_2PI * torch.exp(-0.5 * d * d)
    return (F - K) * cdf + std * pdf


def _tenor_index(model, t):
    """Count of tenor starts T_0..T_{N-1} strictly before t (1e-6 tolerance), as a 0-d tensor."""
    return torch.sum(model.T[:-1] < t - 1e-6)


def _swap_and_deflated_annuity(F_active, taus_active):
    """Pathwise par swap rate, deflated annuity and guard trigger on the active forwards.

    F_active (n_paths, m) holds F_k..F_{N-1} at the date, taus_active (m,) the matching τ. Returns
    (S, A_deflated, onepvf_min): S = (1 − P(T_k,T_N))/A(T_k), A_deflated = A(T_k)/P(T_k,T_N) (the
    terminal-measure value of the annuity, P clamped at 1e-12), and the per-path min over forwards
    of (1+τF), the deflator-control guard trigger.
    """
    one_plus_tauF = 1.0 + taus_active * F_active
    df_ratios = 1.0 / one_plus_tauF
    P_Tk_T = torch.cumprod(df_ratios, dim=1)
    P_Tk_TN = P_Tk_T[:, -1]
    A_t = torch.sum(taus_active * P_Tk_T, dim=1)
    S_t = (1.0 - P_Tk_TN) / A_t
    # deflated annuity A(T_k)/P(T_k,T_N): the terminal-measure value of the swap annuity
    A_t_deflated = A_t / torch.clamp(P_Tk_TN, min=1e-12)

    # per-path min over forwards of (1+τF), the deflator-control guard trigger: a
    # sign-flip (1+τF ≤ 0) collapses P_Tk_TN and blows up A_t_deflated (the CV control D).
    return S_t, A_t_deflated, one_plus_tauF.min(dim=1).values


def _bermudan_gaussian_cv(model, time_grid, anchor_step, n_paths, seed, strike):
    """Exact-Gaussian co-terminal swaption control variate for the Bermudan deflated cashflow.

    Mirrors the calibration CV (pricers.mc_rough_sabr_pricer): a flat-unit-vol Gaussian swap-rate
    proxy S_proxy = S₀ + ξ with law exactly N(S₀, σ_S²), since ξ is a closed-form linear functional
    of the same Gaussian draws (no SDE integrated, so no discretisation residual). Its Bachelier
    mean is the exact E[pay_cv]; with that mean on-tape and β detached upstream, the CV term has
    zero mean in price and greeks. It captures the directional (S−K)⁺ variance only (deflator frozen
    at A0d); the stochastic terminal deflator 1/P(T,T_N) has its own control D in bermudan_pricer.

    Returns (pay_cv, true_cv_mean): pay_cv on-tape via S₀,A0d(F0) and ξ,σ_S(α); true_cv_mean exact.
    """
    device, dtype = model.device, model.dtype
    dt = float(time_grid[1] - time_grid[0])
    T_a = float(time_grid[anchor_step])

    # co-terminal swap = the swap fixing at the anchor date, running to T_N
    k0 = int(_tenor_index(model, T_a).item())
    taus = model.tau[k0:]

    # on-tape forward par rate S₀ and time-0 deflated annuity A0d (frozen, constant across paths)
    F0_swap = model.F0[k0:]
    P_fwd = torch.cumprod(1.0 / (1.0 + taus * F0_swap), dim=0)
    A_fwd = torch.sum(taus * P_fwd)
    P_TN = P_fwd[-1]
    S0 = (1.0 - P_TN) / A_fwd
    A0d = A_fwd / torch.clamp(P_TN, min=1e-12)

    # swap-rate deltas w_k = ∂S/∂F_k|F0 on a separate leaf, so the inner grad cannot free or disturb
    # the main deflated_cf→price subgraph; the detached direction builds no second-order graph.
    # enable_grad: the OOS pilot runs under no_grad, but this inner derivative still needs a tape.
    with torch.enable_grad():
        F0_w = model.F0.detach().clone().requires_grad_(True)
        Fw = F0_w[k0:]
        Pw = torch.cumprod(1.0 / (1.0 + taus * Fw), dim=0)
        Sw = (1.0 - Pw[-1]) / torch.sum(taus * Pw)
        w = torch.autograd.grad(Sw, F0_w)[0][k0:].detach()

    # exact Gaussian std: σ_S² = T_a · ‖loadings_ratesᵀ(w∘α)‖² (α on-tape, so ∂σ_S/∂α flows); rate k
    # uses loadings row k+1, so the swap block is rows k0+1: .
    a = w * model.alphas[k0:]
    g = model.loadings[k0 + 1 :, :].T @ a  # (n_factors,)
    sigma_S = torch.sqrt(torch.clamp(T_a * torch.sum(g * g), min=1e-24))

    # regenerate the increments the sim used (seed-matched), so ξ is correlated with F_paths. Only the
    # full draw reproduces the simulation's stream (Sobol + bridge, or PRNG above the Sobol cap);
    # reduce it to the cumulative shock and free it at once.
    dz_all = generate_sobol_bridge_increments(
        n_paths, len(time_grid) - 1, model.n_factors, dt, device, dtype, seed=seed
    )
    cum_dz = (
        dz_all[:, :anchor_step, :].sum(dim=1).detach()
    )  # (n_paths, n_factors), exactly N(0, T_a·LLᵀ)
    del dz_all
    xi = cum_dz @ g  # on-tape via g(α); ξ ~ N(0, σ_S²) exactly
    pay_cv = A0d * torch.clamp(S0 + xi - strike, min=0.0)
    true_cv_mean = A0d * _bachelier_total(S0, strike, sigma_S)
    return pay_cv, true_cv_mean


def _terminal_deflator_mean(model, last_k_idx):
    """Exact martingale mean m_def = E[D] of the terminal deflated annuity D = A(T_last)/P(T_last,T_N).

    Under Q^{T_N}, P(t,T_k)/P(t,T_N) is a martingale, so E[D] = Σ_k τ_k P(0,T_k)/P(0,T_N) =
    A(0)/P(0,T_N) exactly. Built on-tape from the live F0 for the co-terminal swap fixing at
    `last_k_idx`: the deflator control's analytic mean, the analogue of A0d for that fixing (not
    the Gaussian CV's anchor-date A0d).
    """
    taus_term = model.tau[last_k_idx:]
    F0_term = model.F0[last_k_idx:]
    P_fwd = torch.cumprod(1.0 / (1.0 + taus_term * F0_term), dim=0)
    return torch.sum(taus_term * P_fwd) / torch.clamp(P_fwd[-1], min=1e-12)


def _winsorize(x, q):
    """Two-sided winsorisation: clamp x to its [q, 1−q] empirical quantiles. β-estimation only."""
    lo = torch.quantile(x, q)
    hi = torch.quantile(x, 1.0 - q)
    return torch.clamp(x, min=lo, max=hi)


def _fit_cv_beta(deflated_cf, pay_cv, D_stack, cv_flag_stack, q):
    """(1+n_ex)-control OLS β-vector for vr = X − β₀(Y_gauss−m₀) − Σᵢ βᵢ(Dᵢ−mᵢ).

    Stacks [Y_gauss, D_1, …, D_last] and regresses the detached, winsorised pilot tensors on X.
    Winsorising only the β fit (never the applied price) leaves the estimator unbiased: β is
    detached and every control mean is analytic and on-tape, so E[vr]=E[X] and ∂_θE[vr]=∂_θE[X]
    for any constant β.

    A pilot path flagged on any date (min_j(1+τF) ≤ ε, a D sign-flip blow-up) is dropped from the
    joint OLS: one invalid row poisons the whole β vector, and per-column winsorisation does not
    make the row jointly sane. With one exercise date and no flagged path this recovers the
    two-control fit exactly; the application side keeps the guard's per-component zeroing (see
    bermudan_pricer).

    Returns (beta, cond): beta detached, length 1+n_ex; cond = condition number of the centered
    winsorised design (adjacent Dᵢ are collinear; reported, not fixed).
    """
    keep = ~cv_flag_stack.any(dim=1)  # drop rows flagged at any date
    X = _winsorize(deflated_cf.detach()[keep], q)
    cols = [_winsorize(pay_cv.detach()[keep], q)]
    for j in range(D_stack.shape[1]):
        cols.append(_winsorize(D_stack.detach()[keep, j], q))
    Ycen = torch.stack([c - c.mean() for c in cols], dim=1)  # (n_keep, 1+n_ex)
    Xcen = (X - X.mean()).unsqueeze(1)  # (n_keep, 1)
    beta = torch.linalg.lstsq(Ycen, Xcen).solution.squeeze(1)  # (1+n_ex,)
    cond = float(torch.linalg.cond(Ycen))
    return beta.detach(), cond


def bermudan_pricer(
    model,
    trade_specs,
    n_paths,
    time_grid,
    use_checkpoint=False,
    freeze_drift=True,
    return_weights=False,
    precomputed_weights=None,
    seed=42,
    use_control_variate=False,
    return_beta=False,
    precomputed_beta=None,
    cv_anchor_step=None,
    return_cv_tensors=False,
    driver_type="sobol",
):
    """
    LSM Bermudan payer swaption with AAD Greeks, priced on the exercise boundary fitted to its own
    paths unless `precomputed_weights` is given (oos_bermudan_pricer is the out-of-sample path).

    `use_control_variate` attaches a (1+n_ex)-control exact-mean CV to the final per-path deflated
    cashflow X: (i) the exact-Gaussian co-terminal swaption Y_gauss (directional (S−K)⁺ variance,
    mean A0d·Bachelier) and (ii) the per-exercise-date deflator vector {D_i = A(T_i)/P(T_i,T_N)}
    on all paths, each with on-tape analytic mean m_i = A(0;T_i→T_N)/P(0,T_N). The vector targets
    the heavy-tailed deflator-coupling variance at the realised stopping time, where a single terminal
    control is inert. β is a (1+n_ex)-vector OLS on the winsorised, guard-row-excluded pilot
    tensors (return_beta), frozen and applied on the main run (precomputed_beta). Every mean is
    analytic and on-tape and β is detached, so price and greeks stay unbiased. At n_ex=1
    (European) the vector reduces to the 2-control CV (Gaussian swaption and deflator). Off by default.

    `driver_type` is passed to simulate_forward_curve: "sobol" (default) is production, "prng"
    drives the PRNG oracle in tests/test_fmm_statistical_bias.py.
    """
    log_progress("Action", "Starting Accelerated Bermudan Pricing...", 0)
    log_progress("Simulation", f"Generating {n_paths} FMM paths...", 1)

    strike = torch.tensor(trade_specs["Strike"], device=model.device, dtype=torch.float64)
    ex_dates = torch.tensor(trade_specs["Ex_Dates"], device=model.device, dtype=torch.float64)

    ex_steps = [torch.argmin(torch.abs(time_grid - d)).item() for d in ex_dates]
    n_ex = len(ex_steps)
    if use_control_variate and driver_type != "sobol":
        # the Gaussian control regenerates the Sobol stream; on other drivers it is uncorrelated
        raise ValueError("use_control_variate requires driver_type='sobol'")
    if use_control_variate and n_ex > 1:
        warnings.warn(
            f"deflator control variate on a {n_ex}-exercise Bermudan carries an uncorrected "
            "frozen-drift bias; it is safe only for n_ex=1",
            UserWarning,
            stacklevel=2,
        )

    # only the exercise-date slices are read: F_ex[:, i] = F(t_{ex_steps[i]})
    F_ex = model.simulate_forward_curve(
        n_paths,
        time_grid,
        freeze_drift=freeze_drift,
        use_checkpoint=use_checkpoint,
        seed=seed,
        driver_type=driver_type,
        steps=ex_steps,
    )

    # terminal cashflow at the last exercise date
    last_step = ex_steps[-1]
    last_t = time_grid[last_step].item()
    last_k_idx = _tenor_index(model, last_t).int().item()

    S_term, A_deflated_term, onepvf_min_term = _swap_and_deflated_annuity(
        F_ex[:, n_ex - 1, last_k_idx:], model.tau[last_k_idx:]
    )

    intrinsic_terminal = torch.clamp(S_term - strike, min=0.0)
    deflated_cf = intrinsic_terminal * A_deflated_term

    # per-date CV references in date order [D_1, …, D_last], reused from the exercise loop (the
    # no-CV path allocates nothing); the terminal date fills the last slot.
    if use_control_variate:
        D_list = [None] * n_ex
        onepvf_list = [None] * n_ex
        k_idx_list = [None] * n_ex
        D_list[n_ex - 1] = A_deflated_term
        onepvf_list[n_ex - 1] = onepvf_min_term
        k_idx_list[n_ex - 1] = last_k_idx

    saved_weights = {}

    # start at n_ex-2: terminal payoff already set; regression at maturity is degenerate
    for i in range(n_ex - 2, -1, -1):
        ex_step = ex_steps[i]
        t_now = time_grid[ex_step].item()

        k_idx = _tenor_index(model, t_now).int().item()
        F_t = F_ex[:, i, k_idx:]
        taus = model.tau[k_idx:]

        S_t, A_t_deflated, onepvf_min_i = _swap_and_deflated_annuity(F_t, taus)
        intrinsic_deflated = torch.clamp(S_t - strike, min=0.0) * A_t_deflated

        if use_control_variate:
            # keep the on-tape per-date deflator + guard trigger for the CV vector
            D_list[i] = A_t_deflated
            onepvf_list[i] = onepvf_min_i
            k_idx_list[i] = k_idx

        # normalise by strike to condition the lstsq basis
        S_scaled = S_t / strike
        reg_basis = torch.stack([torch.ones_like(S_scaled), S_scaled, S_scaled**2], dim=1)

        itm = intrinsic_deflated > 0

        if precomputed_weights is not None:
            # main run
            reg_weights = precomputed_weights.get(i, None)
            if reg_weights is not None:
                continuation = reg_basis @ reg_weights
                exercise_now = itm & (intrinsic_deflated > continuation)
            else:
                # strict non-anticipation: no boundary known, don't exercise
                exercise_now = torch.zeros_like(itm, dtype=torch.bool)
        else:
            # pilot run
            if itm.sum() > 50:
                # detached: the LSM fit must stay off the AAD tape
                reg_weights = torch.linalg.lstsq(
                    reg_basis[itm].detach(), deflated_cf[itm].detach()
                ).solution
                if return_weights:
                    saved_weights[i] = reg_weights.detach()
                continuation = reg_basis @ reg_weights
                exercise_now = itm & (intrinsic_deflated > continuation)
            else:
                if return_weights:
                    saved_weights[i] = None
                exercise_now = torch.zeros_like(itm, dtype=torch.bool)

        deflated_cf = torch.where(exercise_now, intrinsic_deflated, deflated_cf)

    if use_control_variate:
        anchor = ex_steps[0] if cv_anchor_step is None else cv_anchor_step
        pay_cv, true_cv_mean = _bermudan_gaussian_cv(
            model, time_grid, anchor, n_paths, seed, strike
        )
        # per-date deflators D_stack and analytic means m_stack (both on-tape) plus the guard
        # masks, in date order [T_1, …, T_last]; m_i is the co-terminal deflator mean fixing at T_i.
        D_stack = torch.stack(D_list, dim=1)  # (n_paths, n_ex)
        m_stack = torch.stack([_terminal_deflator_mean(model, k) for k in k_idx_list])  # (n_ex,)
        cv_flag_stack = torch.stack(onepvf_list, dim=1) <= config.CV_DEFLATOR_GUARD_EPS

    if return_weights:
        if return_beta:
            # pilot: β is detached and frozen for the OOS main run; cond is the design condition
            # number (adjacent-date collinearity diagnostic).
            beta, cond = _fit_cv_beta(
                deflated_cf, pay_cv, D_stack, cv_flag_stack, config.CV_PILOT_WINSOR_Q
            )
            return saved_weights, beta, cond
        return saved_weights

    log_progress("AAD", "Computing Greeks via AAD backward pass...", 0)

    P0_T = torch.cumprod(1.0 / (1.0 + model.tau * model.F0), dim=0)
    P0_TN = P0_T[-1]

    if use_control_variate and return_cv_tensors:
        # test-only hook (test_fmm_statistical_bias._per_path_diag): X, Y_gauss and its exact mean,
        # D_stack (n_paths, n_ex), m_stack (n_ex,), and the guard masks (n_paths, n_ex). The
        # consumer recombines them with the pilot β exactly as the branch below does.
        return (
            deflated_cf.detach(),
            pay_cv.detach(),
            true_cv_mean.detach().item(),
            D_stack.detach(),
            m_stack.detach(),
            cv_flag_stack.detach(),
        )

    if use_control_variate and precomputed_beta is not None:
        # vr = X − β₀(Y_gauss − m₀) − Σᵢ βᵢ(Dᵢ − mᵢ). The Dᵢ − mᵢ terms stay on-tape: detaching Dᵢ
        # while mᵢ is on-tape would inject −βᵢ∂_θmᵢ into the gradient and bias the greeks. The
        # guard zeroes each βᵢ deflator term on its own flagged (sign-flip) paths only; X and the β₀
        # Gaussian term are untouched.
        def_terms = precomputed_beta[1:] * (D_stack - m_stack)  # (n_paths, n_ex)
        def_terms = torch.where(cv_flag_stack, torch.zeros_like(def_terms), def_terms)
        vr = deflated_cf - precomputed_beta[0] * (pay_cv - true_cv_mean) - def_terms.sum(dim=1)
        return P0_TN * torch.mean(vr)

    return P0_TN * torch.mean(deflated_cf)


def oos_bermudan_pricer(
    model,
    trade_specs,
    n_paths_main=16384,
    n_paths_pilot=4096,
    time_grid=None,
    use_checkpoint=False,
    freeze_drift=True,
    seed=42,
    verbose=False,
    use_control_variate=False,
    cv_anchor_step=None,
    driver_type="sobol",
):
    """
    Out-of-sample Bermudan: an independent pilot (seed+999) fits the LSM weights and the main run
    prices on them, so the boundary carries no foresight from the main paths.

    With `use_control_variate`, the pilot also returns the (1+n_ex)-control β, passed to the main
    run as oos_mc_rough_sabr_pricer passes beta_pilot (cond is dropped here).

    `driver_type` goes to both runs: "sobol" (default) is production, "prng" drives the PRNG
    oracle in tests/test_fmm_statistical_bias.py.
    """
    if verbose:
        print(
            f"      [OOS Bermudan] Starting Pilot Run (N={n_paths_pilot}, seed={seed + 999})...",
            end="",
            flush=True,
        )
        pilot_t_start = time.time()

    # the pilot only fits LSM weights / CV β (all detached): no tape needed for it
    with torch.no_grad():
        pilot_out = bermudan_pricer(
            model,
            trade_specs,
            n_paths=n_paths_pilot,
            time_grid=time_grid,
            use_checkpoint=use_checkpoint,
            freeze_drift=freeze_drift,
            return_weights=True,
            precomputed_weights=None,
            seed=seed + 999,
            use_control_variate=use_control_variate,
            return_beta=use_control_variate,
            cv_anchor_step=cv_anchor_step,
            driver_type=driver_type,
        )
    if use_control_variate:
        pilot_weights, beta_pilot, _ = pilot_out
    else:
        pilot_weights, beta_pilot = pilot_out, None

    if verbose:
        print(f" Done ({time.time() - pilot_t_start:.2f}s).", flush=True)
        print(
            f"      [OOS Bermudan] Starting Main Run  (N={n_paths_main}, seed={seed})...",
            end="",
            flush=True,
        )
        main_t_start = time.time()

    main_price = bermudan_pricer(
        model,
        trade_specs,
        n_paths=n_paths_main,
        time_grid=time_grid,
        use_checkpoint=use_checkpoint,
        freeze_drift=freeze_drift,
        return_weights=False,
        precomputed_weights=pilot_weights,
        seed=seed,
        use_control_variate=use_control_variate,
        precomputed_beta=beta_pilot,
        cv_anchor_step=cv_anchor_step,
        driver_type=driver_type,
    )

    if verbose:
        print(f" Done ({time.time() - main_t_start:.2f}s).", flush=True)

    return main_price
